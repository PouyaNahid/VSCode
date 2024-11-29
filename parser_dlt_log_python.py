"""
DLT Parser
"""
import os
import platform
from datetime import datetime
from subprocess import run
import re

import pandas as pd
import slash


REMOTE_ECUS_LIST = [258, 160, 264, 265, 164, 266, 106, 8, 281]

TEST_TYPE = 'Physical'   # Physical | Functional
INTERNAL_TESTER_ECUS = [258, 264, 266]  # 102, 108, 10A: connected to internal terster
SEARCH_KEYWORDS = {'Physical': {'KW_DGW_GET_RQST': ['DoipUdsRouterInstanceId method_name= IndicateMessage call_id'],
                                # + 3712 264 which is 0xE80 0x108
                                'KW_DGW_SEND_RQST': ['frameSender'],
                                # + External Tester ADDR(hex) + ECU addr(hex) like: 0E 80 01 08 which is 0xE80, 0x108
                                # ['frameSender', 0E 80 01 08]  or
                                # + Internal Tester ADDR(hex) + ECU addr(hex) like: 0F 00 01 02 which is 0xF00, 0x102
                                # ['frameSender', 0F 00 01 02]
                                'KW_DGW_GET_RESP': ['External DiagMsg(8001)'],
                                # + SA: ECU ADDR + TA: Tester ADDR like SA: 108 + TA: E80
                                # ['External DiagMsg(8001)', SA: 108, TA: E80 ]
                                # ['External DiagMsg(8001)', SA: 108, TA: F00 ]
                                'KW_DGW_SEND_RESP': ['sendTransmitMessage']
                                # + SA: ECU ADDR like SA: 108
                                # ['sendTransmitMessage', SA: 108]
                                },
                   'Functional': {'KW_DGW_GET_RQST': ['SA: E80 TA: E400 udsRequestPayload(hex)'],
                                  'KW_DGW_SEND_RQST': ['frameSender'],
                                  # + External Tester ADDR Functional ADDR + TA: ECU addr like: 0E 80 E4 00 + TA: A4
                                  # ['frameSender', 0E 80 E4 00, TA: A4] or
                                  # + Internal Tester ADDR(hex) + ECU addr(hex) like: 0F 00 E4 00 + TA: 10A
                                  # ['frameSender', 0F 00 E4 00, TA: 10A]
                                  'KW_DGW_GET_RESP': ['External DiagMsg(8001)'],
                                  # + SA: ECU ADDR + TA: Tester ADDR like SA: 108 + TA: E80
                                  # ['External DiagMsg(8001)', SA: 108, TA: E80 ]
                                  # ['External DiagMsg(8001)', SA: 108, TA: F00 ]
                                  'KW_DGW_SEND_RESP': ['sendTransmitMessage']
                                  # + SA: ECU ADDR + TA: Functional ADDR like SA: 108 + TA: E400
                                  # ['sendTransmitMessage', SA: 10A TA: E400]
                                  }}
SEARCH_KEYWORDS = {'Physical': {

    'KW_DM_INDICATE': ['IndicateMessage corrId:'],  # S1
    # + SA: Tester ADDR + TA: ECU ADDR like: SA: E80 TA: 119
    # ['IndicateMessage corrId:', SA: E80 TA: 119]

    'KW_DM_BACK_INDICATE': ['IndicateMessage corrId:', 'isFunc: false'],  # S2
    # + TA: ECU ADDR like TA: 119
    # ['IndicateMessage corrId:, isFunc: false, TA: 119]

    'KW_DM_HANDLE': ['HandleMessage corrId:'],  # S3,
    # + SA: Tester ADDR TA: ECU ADDR like SA: E80 TA: 119
    # [HandleMessage corrId:, SA: E80 TA: 119]

    'KW_DGW_TO_ECU': ['send External corrId:', '02 FD 80 01'],  # S3,
    # + TA: ECU ADDR like: TA: 119
    # [send External corrId:, TA: 119, 02 FD 80 01]

}}

DLT_LOG_FILE_NAME = 'AppET.dlt'
IT_DISCRETE_HEX_ADDR = '0F 00'
ET_DISCRETE_HEX_ADDR = '0E 80'
FUNC_DISCRETE_HEX_ADDR = 'E4 00'

IT_HEX_ADDR = 'F00'
ET_HEX_ADDR = 'E80'
FUNC_HEX_ADDR = 'E400'


def convert_decimals_to_hex(decimal: int) -> list[str] | str:
    """
    Convert each decimal to hexadecimal.
    Check if input is list or not. also upper()[2:] keeps uppercase mode
    without 0X part of hex value. e.g.: 0xfa -> FA
    """
    if isinstance(decimal, list):
        hex_value = [hex(number).upper()[2:] for number in decimal]
    else:
        hex_value = hex(decimal).upper()[2:]
    return hex_value


def add_statistics_to_reshaped_dataframe(data):
    """
    Calculate the min, average, and max for each ECU in a reshaped DataFrame.

    Parameters:
        data (dict): A dictionary containing the reshaped DataFrame data.

    Returns:
        dict: A dictionary containing the calculated statistics.
    """
    crop_df = None
    df = pd.DataFrame(data)

    # Create a new DataFrame excluding the 'Iteration' column
    if 'Iteration' in df:
        crop_df = df.drop(columns=['Iteration'])

    if crop_df is None:
        crop_df = df

    # Calculate the min, average, and max for each ECU
    min_values = crop_df.min().astype(int)
    average_values = crop_df.mean().astype(int)
    max_values = crop_df.max().astype(int)
    median_values = crop_df.median().astype(int)

    # Create a new dictionary with the required structure
    statistic_result = {'Item': list(crop_df.columns),
                        'Min': list(min_values),
                        'Average': list(average_values),
                        'Median': list(median_values),
                        'Max': list(max_values)}
    return statistic_result


class ReportGenerator:   # pylint: disable=R0913, R0914, R0902
    """ Prepare results and concatenate different columns of result, export
    data into spreadsheets and create plot (charts) """

    def __init__(self):
        self.to_excel_data = None
        self.representative_responses = None
        self.total_responses = None
        self.sheet = None
        self.cur_time = None
        self.available_test_cases = ['single', 'dtc', 'random']
        self.file_ext = '.xlsx'
        if 'wind' in platform.system().lower():
            self.log_dir = os.path.dirname(__file__) + '\\log\\'
        elif 'linux' in platform.system().lower():
            self.log_dir = os.path.dirname(__file__) + '/log/'
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def generate_raw_and_statistics_report(self, data,   # pylint: disable=R0913, R0914, R0917
                                           uds_request,
                                           address_type,
                                           more_info,
                                           start_time):
        """    Generate raw and statistics report for the given data.

        Parameters:
        - data: The data to generate the report from.
        - ecu_address: The ECU address.
        - uds_request: The UDS request used to generate the data.
        - address_type: The addressing type used (physical or functional).
        - more_info: Additional information for the report.
        - start_time: The start time of the test.

        Generates a raw data report and a statistics report for the given data. The raw data report is generated
        if the data is not empty, and the statistics report is generated if the statistics data is not empty.
        The reports are created using the create_excel_record function, with different sheet names
        based on the address type and ECU address.
    """
        if data:
            self.create_excel_record(timestamp_recorder=data,
                                     uds_request=uds_request,
                                     address_type=f'{address_type}_Raw',
                                     more_info=more_info,
                                     start_time=start_time)

        statistic_data = add_statistics_to_reshaped_dataframe(data)
        if statistic_data:
            self.create_excel_record(timestamp_recorder=statistic_data,
                                     uds_request=uds_request,
                                     address_type=f'{address_type}_Statistics',
                                     more_info=more_info,
                                     start_time=start_time,
                                     axis_labels=['ECU ID | Sbbfunction', 'ResponseTime(ms)'])

    def prepare_result(self, prf_test_results):
        """ Prepare and grooming columns """
        self.total_responses = []
        self.representative_responses = []
        response_time_list = []
        ctr = 0
        prf_test_results.sort(key=lambda x: x[0])  # sort total response based on ID
        # Extract the fourth element and access the first element
        request_id = [sublist[0] for sublist in prf_test_results]

        # Count occurrences
        ids_repetition = {}
        for did in request_id:
            ids_repetition[did] = ids_repetition.get(did, 0) + 1

        for response in prf_test_results:
            temp = [hex(response[0]).upper(), response[1]]  # [0x603A, ctr]
            response_time = (response[3] - response[2]) * 1000  # millisecond
            response_time_list.append(round(response_time, 2))
            temp.append(round(response_time, 2))  # [0x603A, ctr, time]
            hex_response = convert_decimals_to_hex(response[5])
            temp.append(len(hex_response))  # [0x603A, ctr, time, length]
            temp.append(hex_response)  # [0x603A, ctr, time, length, resp_hex]
            self.total_responses.append(temp)

            ctr += 1
            repeat = ids_repetition[response[0]]
            if ctr == repeat:
                temp = []
                max_response_time = max(response_time_list)
                min_response_time = min(response_time_list)
                mean_response_time = sum(response_time_list) / len(response_time_list)
                temp.append(f'{hex(response[0]).upper()}_{len(hex_response)}')  # [ID_Length]
                temp.extend([round(min_response_time, 2), round(mean_response_time, 2), round(max_response_time, 2)])
                # [ID_Length, min_time, mean_time, max_time]
                temp.extend([len(hex_response), repeat])
                # [ID_Length, min_time, mean_time, max_time, length, total_repeat]
                self.representative_responses.append(temp)
                response_time_list = []
                ctr = 0
        self.representative_responses.sort(key=lambda x: (x[4]))  # sort representative response based on payload length

    def export_excel_report(self, sheet, total_chart: bool = True):
        """ Export Data to Excel File"""
        self.sheet = sheet
        self.cur_time = self.get_time()
        self._generate_excel_summary_report()
        self._generate_excel_total_report(total_chart)

    def _generate_excel_summary_report(self):
        show_file_header = ['DID_Length',
                            'Min Response Time (ms)',
                            'Mean Response Time (ms)',
                            'Max Response Time (ms)',
                            'Data Length (byte)',
                            'Test repeat ']
        columns_index_to_chart = [1, 2, 3]  # Index of columns go to be shown
        chart_type = 'column'
        chart_label_xy = ['DID_Length', 'Response Time (ms)']

        summary_file_path = self.log_dir + self.sheet + '_chart_' + self.cur_time + self.file_ext
        self.write_to_excel_simple_data(file_name=summary_file_path,
                                        data=self.representative_responses,
                                        header=show_file_header,
                                        sheetname=self.sheet,
                                        chart_type=chart_type,
                                        columns_indices=columns_index_to_chart,
                                        axis_labels=chart_label_xy)
        # self._write_to_excel(summary_file_path, show_file_header, chart_type, plot_index, chart_label_xy)

    def _generate_excel_total_report(self, total_chart: bool):
        total_file_header = ['Title', 'Iteration', 'Response Time(ms)', 'Data Length (byte)', 'Data Value']
        if total_chart:
            columns_index_to_chart = [2]
            chart_type = 'line'
        else:
            columns_index_to_chart = []
            chart_type = ''

        total_file_path = self.log_dir + self.sheet + '_total_result_' + self.cur_time + self.file_ext
        self.to_excel_data = self.total_responses
        self.write_to_excel_simple_data(file_name=total_file_path,
                                        data=self.total_responses,
                                        header=total_file_header,
                                        sheetname=self.sheet,
                                        chart_type=chart_type,
                                        columns_indices=columns_index_to_chart)

    def get_time(self):
        """ Return current Time and Date"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def create_excel_record(self,       # pylint: disable=R0913, R0914, R0917
                            timestamp_recorder,
                            uds_request,
                            address_type,
                            more_info,
                            start_time,
                            axis_labels=None):
        """
        Create an Excel record based on the timestamp recordings, UDS request, and addressing type.

        Parameters:
        - timestamp_recorder: A dictionary containing timestamp recordings.
        - uds_request: The UDS request for which the record is created.
        - address_type: The type of addressing used (physical or functional).

        Creates an Excel record with the provided timestamp recordings, formatted UDS request,
        and addressing type information. The Excel file is saved in the log directory with a specific filename.
        """
        formatted_request = ' '.join([f'{x:02X}' for x in uds_request])
        report_charectrestic = address_type.title().replace(' ', '')

        chart_title = f'[{formatted_request}]: {report_charectrestic} | {more_info}: \nResponseTime - RequestTime'
        parsing_file_path = self.log_dir + f'{report_charectrestic}_' + start_time + self.file_ext

        chart_type = 'column' if 'Statistic' in address_type else 'line'

        self.write_to_excel_simple_data(file_name=parsing_file_path,
                                        data=timestamp_recorder,
                                        sheetname=report_charectrestic,
                                        chart_type=chart_type,
                                        chart_title=chart_title,
                                        axis_labels=axis_labels)

    def write_to_excel_simple_data(self,   # pylint: disable=R0913, R0914, R0917
                                   file_name: str,
                                   data: dict | list,
                                   header: list = None,
                                   sheetname: str = 'Result',
                                   chart_type: str = 'line',
                                   columns_indices: list = None,
                                   category_column_indx: int = 0,
                                   chart_title: str = None,
                                   axis_labels: list[str] = None,
                                   export_html: bool = False):
        """ Write simple data to an Excel file and optionally create a chart.

        Parameters:
            - file_name: The name of the Excel file.
            - data: The data to write (dictionary or list).
            - header: Header for list data.  Required if `data` is a list.
            - sheet_name: Worksheet name (default: 'Result').
            - chart_type: Chart type ('line', 'bar', etc.).
            - columns_indices: Columns to include in chart (default: all).
            - category_column_index: Index of category column for chart (default: 0).
            - chart_title: Chart title (default: sheet name).
            - axis_labels: X and Y axis labels (default: ['Iteration', 'Response Time (ms)']).
            - export_html: Export to HTML as well (default: False).

        Raises:
            ValueError: If data and header are incorrectly specified.

        Writes the data to an Excel file with the specified name and worksheet name.
        If the data is a dictionary, it is written as is. If data is a list, header list is used to define column names.
        If specified, a chart of the given type is created using the data in a specified columns and added to worksheet.
        If export_html is True, the Excel file is also exported to an HTML file with the same name.
        """

        assert not (isinstance(data, dict) and header is not None), \
            "Internal Error: Unexpected header with dictionary data."
        assert not (isinstance(data, list) and header is None), \
            "Internal Error: Missing header with list data."

        df = pd.DataFrame(data, columns=header) if isinstance(data, list) else pd.DataFrame(data)

        axis_labels = axis_labels or ['Iteration', 'Response Time (ms)']

        with pd.ExcelWriter(file_name, engine='xlsxwriter', datetime_format='mmm d yyyy hh:mm:ss') as writer:
            df.to_excel(writer, sheet_name=sheetname, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheetname]
            worksheet.autofit()
            # chart_type: Area, Bar, Column, Line, Pie, Doughnut, Scatter, Stock, Radar
            num_rows, num_cols = df.shape
            columns_indices = columns_indices or list(range(1, num_cols))  # Use all columns if none specified.

            # categories: Refers to the X-axis labels (usually the first column of data).
            # values: Refers to the Y-axis values (the data points to be plotted).
            # line: Specifies formatting options for the series (e.g., line color).
            # name: [sheet, cel_row, cel_col],
            # categories:[sheet_name,start_row,start_clo,stop_row, stop_col]
            # values:[sheet_name,start_row,start_clo,stop_row,stop_col]

            # Configure the chart axes, Set an Excel chart style, if there is any chart to plot.
            if columns_indices:    # Only create chart if columns are specified.
                chart = workbook.add_chart({"type": chart_type})
                for col_index in columns_indices:
                    chart.add_series({
                        'name': [sheetname, 0, col_index],
                        'categories': [sheetname, 1, category_column_indx, num_rows, category_column_indx],
                        'values': [sheetname, 1, col_index, num_rows, col_index]})

                chart.set_x_axis({"name": f"{axis_labels[0]}"})
                chart.set_y_axis({"name": f"{axis_labels[1]}", "major_gridlines": {"visible": True}})

                chart.set_title({'name': chart_title or sheetname.replace("_", " ", 2).title})
                # Set width and height in pixels
                chart.set_size({'width': 800, 'height': 600})
                # Insert the chart into the worksheet
                worksheet.insert_chart(num_rows + 2, 1, chart)

def get_ecu_hex_address(ecu: int) -> str:
    """
        Converts an ECU identifier to a hexadecimal string representation.

        Parameters:
            ecu (int): The ECU identifier.

        Returns:
            str: The hexadecimal string representation of the ECU ID.
    """
    return f'{ecu:02X}' if ecu < 0xFF else f'{ecu:X}'


def int_to_two_byte_string(value: int) -> str:
    """
        Convert an integer value to a two-byte string representation.

        Parameters:
        - value (int): The integer value to convert to a two-byte string.

        Returns:
        - str: A two-byte string representation of the integer value in hexadecimal format,
            with each byte separated by a space. Example: '01 13' for the value 275.

        Raises:
        - ValueError: If the input value is not within the range of 0 to 65535.
    """

    # Ensure the value is within the range of 0 to 65535 (16-bit integer)
    if not 0 <= value <= 0xFFFF:
        raise ValueError("Value out of range. Must be between 0 and 65535.")

    # Convert the integer to a 2-byte array in big-endian order
    byte_array = value.to_bytes(2, 'big')

    # Convert each byte to a two-character hexadecimal string and join with a space
    byte_string = ' '.join(f"{byte:02X}" for byte in byte_array)  # Eg.: 275(Dec)=0x113 -> '01 13'

    return byte_string


def calculate_time_difference_ms(start_time: list[str], stop_time: list[str]) -> int:
    """
        Calculate the time difference between two time points.

        Parameters:
        - start_time (list[str]): A list containing [hours, minutes, seconds] for the start time.
        - stop_time (list[str]): A list containing [hours, minutes, seconds] for the stop time.

        Returns:
        - int: The time difference between the start and stop times in milliseconds.

        The function calculates the time difference in milliseconds between the start_time and stop_time
        provided as lists of [hours, minutes, seconds]. It first calculates the time difference in seconds
        and then converts it to milliseconds.
    """
    dh = int(stop_time[0]) - int(start_time[0])
    dm = int(stop_time[1]) - int(start_time[1])
    ds = float(stop_time[2]) - float(start_time[2])
    dt = round((dh * 3600 + dm * 60 + ds) * 1000, 2)  # Milisecond

    return dt


def build_et_keyword_filter(entry: str, keyword: str, ecu: int, ecu_discrete_hex_addr: str, ecu_hex_addr: str) -> list:
    """
        Constructs the initial keyword based on the entry type and ECU ID.

        Parameters:
            entry: The entry to query.
            keyword: The search keyword associated with the query.
            ecu: The ECU identifier.
            ecu_discrete_hex_addr: The discrete hexadecimal address of the ECU.
            ecu_hex_addr: The hexadecimal address of the ECU.

        Returns:
            list: The constructed keyword list.
    """

    if 'KW_DGW_GET_RQST' in entry:
        return keyword + [f'3712 {str(int(ecu))}'] if TEST_TYPE == 'Physical' and ecu != 217 else keyword

    if entry in ['KW_DM_INDICATE', 'KW_DM_HANDLE']:
        return keyword + [
            f'SA: {ET_HEX_ADDR} TA: {ecu_hex_addr}'] if TEST_TYPE == 'Physical' and ecu != 217 else keyword

    if entry in ['KW_DM_BACK_INDICATE', 'KW_DGW_TO_ECU']:
        return keyword + [f'TA: {ecu_hex_addr}'] if TEST_TYPE == 'Physical' and ecu != 217 else keyword

    if 'KW_DGW_SEND_RQST' in entry:
        if TEST_TYPE == 'Physical':
            return keyword + [f'{ET_DISCRETE_HEX_ADDR} {ecu_discrete_hex_addr}']

        return keyword + [f'{ET_DISCRETE_HEX_ADDR} {FUNC_DISCRETE_HEX_ADDR}', f'TA: {ecu_hex_addr}']

    if 'KW_DGW_GET_RESP' in entry:
        return keyword + [f'SA: {ecu_hex_addr} TA: {ET_HEX_ADDR}']

    if TEST_TYPE == 'Physical':  # When entry is: KW_DGW_SEND_RESP
        return keyword + [f'SA: {ecu_hex_addr}']

    return keyword + [f'SA: {ecu_hex_addr} TA: {FUNC_HEX_ADDR}']


def build_it_keyword_filter(entry: str, keyword: str, ecu_discrete_hex_addr: str, ecu_hex_addr: str) -> list:
    """
        Constructs the keyword by adding the internal tester address when no matches are found.

        Parameters:
            entry: The entry to query.
            keyword: The search keyword associated with the query.
            ecu_discrete_hex_addr: The discrete hexadecimal address of the ECU.
            ecu_hex_addr: The hexadecimal address of the ECU.

        Returns:
            list: The constructed keyword list with internal tester address.
    """
    if 'KW_DGW_GET_RESP' in entry:
        return keyword + [f'SA: {ecu_hex_addr} TA: {IT_HEX_ADDR}']
    if 'KW_DGW_SEND_RQST' in entry:
        if TEST_TYPE == 'Physical':
            return keyword + [f'{IT_DISCRETE_HEX_ADDR} {ecu_discrete_hex_addr}']

        return keyword + [f'{IT_DISCRETE_HEX_ADDR} {FUNC_DISCRETE_HEX_ADDR}', f'TA: {ecu_hex_addr}']
    return keyword


class DLTParser():
    """This class is defined for Parsing DLT log file and extracet timestamps for predefined DGW Entries."""
    dgw_recv_send_rqst_resp_entry_kws = SEARCH_KEYWORDS[TEST_TYPE]
    dlt = None
    matched_rows = None
    min_recorded_match_sample = 10_000_000
    rp = ReportGenerator()

    def load_dlt_csv_file(self, csv_file: str, seperator: str = ',') -> None:
        """
            Reads the CSV file specified by 'csv_file' into the 'dlt' DataFrame using pandas 'read_csv' method.
            Parameters:
            - csv_file (str): The path to the CSV file to load.
            - separator (str): The separator used in the CSV file (default is ',').
        """
        slash.logger.info(f'Loading CSV file: {csv_file}')
        try:
            self.dlt = pd.read_csv(csv_file, sep=seperator)
            slash.logger.info(f'DLT Log Size: {self.dlt.shape}')
        except FileNotFoundError as err:
            slash.logger.error(f'Error loading CSV file: {err}')

    def filter_csv_by_keywords(self, keywords: list[str], all_keywords: bool = True):
        """
            This method filters the rows in the 'dlt' DataFrame by checking if all the keywords provided
            appear in the 'Payload' column of each row. It updates the 'match_rows' attribute with the filtered rows.

            Parameters:
            - keywords (list[str]): A list of keywords to search for in the 'Payload' column.
        """
        filter_func = all if all_keywords else any  # Cleaner way to select all/any
        self.matched_rows = self.dlt[(self.dlt["Apid"] == 'DGW') &
                                     (self.dlt["Payload"].apply(
                                         lambda x: filter_func(keyword in x for keyword in keywords)))]

        slash.logger.info(f"Keywords:{keywords}, Matched Rows:{len(self.matched_rows)}")
        slash.logger.info(f'Time:\n{self.matched_rows["Time"]} ')
        if not self.matched_rows.empty:
            self.min_recorded_match_sample = min(self.min_recorded_match_sample, len(self.matched_rows))

    def extract_matched_timestamps(self) -> list:
        """
            Extract the time information from the matched rows.

            Returns a list of time values in the format [hh, mm, ss.sss] for each row in the match_rows DataFrame.
            The time information is extracted from the 'Time' column of each row, and the extracted values are
            added to the output list.

            Returns:
            - list: A list containing time values in the format [hh, mm, ss.sss].
        """
        timestamps_corrids = []
        for _, row in self.matched_rows.iterrows():
            # time_hh_mm_ssdotms.append(row['Time'][11:].split(':'))
            timestamp = pd.to_datetime(row['Time'])  # Convert to datetime object
            corrId_match = re.search(r'corrId:\s*(\d+)', row['Payload'])
            if corrId_match:
                corrId = int(corrId_match.group(1))
                timestamps_corrids.append((timestamp, corrId))  # Store both timestamp and corrId
        return timestamps_corrids

    def calculate_delta_time_of_two_entry(self, timestamp_ecu_sample: dict, entry_a: str, entry_b: str):
        """Calculate time differences between entries with matching corrIds and organize into a dictionary."""

        results = {'Iteration': []}  # Initialize the results dictionary with 'Iteration' column

        # Keep track of maximum number of iterations across all ECUs
        max_iterations = 0
        # timestamps_ecu_sample is somthing like:
        # {
        #  258: {
        #       'KW_DGW_GET_RQST': [['11', '28', '05.953127'], ['11', '28', '07.189936']],
        #       'KW_DGW_SEND_RQST': [['11', '28', '05.951632'], ['11', '28', '07.185723']]
        #       },
        #  260: {
        #       'KW_DGW_GET_RQST': [['11', '28', '05.953127'], ['11', '28', '07.189936']],
        #       'KW_DGW_SEND_RQST': [['11', '28', '05.951632'], ['11', '28', '07.185723']]
        #       }
        # }
        for ecu, entries in timestamp_ecu_sample.items():
            entries_a = entries.get(entry_a, [])
            entries_b = entries.get(entry_b, [])

            # Create DataFrames for easier manipulation
            df_a = pd.DataFrame(entries_a, columns=['time', 'corrId']) if entries_a else pd.DataFrame(
                columns=['time', 'corrId'])
            df_b = pd.DataFrame(entries_b, columns=['time', 'corrId']) if entries_b else pd.DataFrame(
                columns=['time', 'corrId'])

            merged_df = pd.merge(df_a, df_b, on='corrId', suffixes=('_a', '_b'))
                
            if not merged_df.empty:
                merged_df['time_diff'] = round((merged_df['time_b'] - 
                                                merged_df['time_a']).dt.total_seconds() * 1000, 2)  # Milliseconds
                results[ecu] = merged_df['time_diff'].tolist()  # Add time differences to ECU column
                max_iterations = max(max_iterations, len(merged_df))

        # Populate the 'Iteration' column and pad shorter lists with None
        results['Iteration'] = list(range(1, max_iterations + 1))
        for ecu in timestamp_ecu_sample:
            if ecu in results:
                results[ecu].extend([None] * (max_iterations - len(results[ecu])))  # Pad with None for missing values
            else:
                results[ecu] = [None] * max_iterations  # Initialize with None if no entries for an ECU

        # Calculate and add aggregate statistics
        results['Iteration'].extend(['Min', 'Median', 'Average', 'Mode', 'Max'])
        for ecu, values in results.items():
            # Exclude the 'Iteration' column from aggregation calculations
            if ecu != 'Iteration' and not all(x is None for x in values):  
                # Create a DataFrame from the ECU's data after filtering None values
                df = pd.DataFrame({ecu: [x for x in values if  x is not None]})  

                results[ecu].append(df[ecu].min())
                results[ecu].append(df[ecu].median())
                results[ecu].append(round(df[ecu].mean(), 2))
                results[ecu].append(df[ecu].mode()[0] if not df[
                    ecu].mode().empty else None)  # Get the first mode or None if the mode series is empty
                results[ecu].append(df[ecu].max())

            elif ecu != 'Iteration':  # Pad with None for ECUs without data
                results[ecu].extend([None] * 5)

        return results

    def get_dgw_entry_ecu_time(self, entry: str, keyword: str, ecu: int) -> None:
        """
            Retrieves DLT log Payload column information related to a specific DGW entry, search keyword
            and ECU ID.

            Parameters:
                entry (str): The entry to query (e.g., 'KW_DGW_GET_RQST', 'KW_DGW_SEND_RQST').
                keyword (str): The search keyword associated with the query.
                ecu (int): The ECU identifier.

            Returns:
                None: The function modifies internal state (e.g., `self.match_rows`).

            Notes:
                - If the entry is 'KW_DGW_GET_RQST', the function constructs a keyword + ET ID(3712) ECU ID.
                - If the entry is 'KW_DGW_SEND_RQST', the function constructs a keyword + external tester
                address(0E 80) and ECU ID.
                - Otherwise, it constructs a keyword + SA: ECU ID.
                - If no matches are found, it adds the internal tester address to the keyword.
        """
        ecu_discrete_hex_addr = int_to_two_byte_string(ecu)
        ecu_hex_addr = get_ecu_hex_address(ecu)

        ultimate_kw = build_et_keyword_filter(entry, keyword, ecu, ecu_discrete_hex_addr, ecu_hex_addr)
        self.filter_csv_by_keywords(ultimate_kw)

        if self.matched_rows.empty:
            ultimate_kw = build_it_keyword_filter(entry, keyword, ecu_discrete_hex_addr, ecu_hex_addr)
            slash.logger.info(f'Internal Tester works for ECU: {ecu_discrete_hex_addr}')
            self.filter_csv_by_keywords(ultimate_kw)

    def init_dlt_csv_file(self) -> str:
        """
            Initialize a CSV file from a DLT log file using the dlt-viewer command line tool.

            Returns:
            - str: The path to the created CSV file.

            Raises:
            - subprocess.SubprocessError: If an error occurs while running the dlt-viewer command.
            - TimeoutError: If the command execution times out.
        """
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dlt_log_file_path = os.path.join(current_dir, DLT_LOG_FILE_NAME)

        csv_file_name = DLT_LOG_FILE_NAME.replace('.dlt', '.csv')
        csv_file_name_path = os.path.join(current_dir, csv_file_name)

        if not os.path.exists(csv_file_name_path):
            print('Convert DLT to CSV file is processing ...')
            cmd = ['dlt-viewer', '-s', '-csv', '-c', f'{dlt_log_file_path}', csv_file_name_path]
            process = run(cmd, capture_output=True, check=True, timeout=250)
            if process.returncode == 0:
                print("Command executed successfully.")
            else:
                print(f"Error running command: {process.stderr}")  # Print the error message
                raise process.stderr
        else:
            print('CSV file already exists.')

        return csv_file_name_path

    def test_dlt_parser_script(self):
        """
            Test is not a test in real, just to be compatible with slash fixture and rules.
            This function calculates the delta time between different entries of DGW.
            It reads the DLT log file which is in CSV format.
            We defined 4 Entries for DGW as:
                T1: Entry1 of DGW: When DGW receives the request from DM (External Tester).
                T2: Entry2 of DGW: When DGW sends the request out to Remote ECU.
                T3: Entry3 of DGW: When DGW gets back the response from Remote ECU.
                T4: Entry4 of DGW: When DGW sends out the response to DM.
        """

        csv_file_name_path = self.init_dlt_csv_file()
        # DGW Entry/Exit Times for each ECU
        time_dgw_entries_ecus = {ecu: {} for ecu in REMOTE_ECUS_LIST}

        # DGW each Entry/Exit Duration for two continues iteration

        self.load_dlt_csv_file(csv_file=csv_file_name_path)
        # We have 4 cases which are the entry and exit point of DGW when gets a request from ET and sends it to ECU,
        # gets response and sends back to ET
        for entry, keyword in self.dgw_recv_send_rqst_resp_entry_kws.items():
            for ecu in REMOTE_ECUS_LIST:
                self.get_dgw_entry_ecu_time(entry, keyword, ecu)

                time_dgw_entries_ecus[ecu][entry] = self.extract_matched_timestamps()

        # delta_t41 = self.calculate_delta_time_of_two_entry(time_dgw_entries_ecus, 'KW_DGW_GET_RQST', 'KW_DGW_SEND_RESP')
        # delta_t21 = self.calculate_delta_time_of_two_entry(time_dgw_entries_ecus, 'KW_DGW_GET_RQST', 'KW_DGW_SEND_RQST')
        # delta_t43 = self.calculate_delta_time_of_two_entry(time_dgw_entries_ecus, 'KW_DGW_GET_RESP', 'KW_DGW_SEND_RESP')
        # KW_DM_INDICATE, KW_DM_BACK_INDICATE, KW_DM_HANDLE, KW_DGW_TO_ECU
        delta_t41 = self.calculate_delta_time_of_two_entry(time_dgw_entries_ecus, 'KW_DM_INDICATE', 'KW_DGW_TO_ECU')
        delta_t21 = self.calculate_delta_time_of_two_entry(time_dgw_entries_ecus, 'KW_DM_INDICATE',
                                                           'KW_DM_BACK_INDICATE')
        delta_t43 = self.calculate_delta_time_of_two_entry(time_dgw_entries_ecus, 'KW_DM_HANDLE', 'KW_DGW_TO_ECU')

        parsing_file_path = f'{self.rp.log_dir}{TEST_TYPE}Parsing_T4_T1_{self.rp.get_time()}{self.rp.file_ext}'
        self.rp.write_to_excel_simple_data(file_name=parsing_file_path, data=delta_t41,
                                        sheetname='DLT_Parse', chart_title='T4 - T1: DGW Send RESP - DGW Get RQST')

        parsing_file_path = f'{self.rp.log_dir}{TEST_TYPE}Parsing_T2_T1_{self.rp.get_time()}{self.rp.file_ext}'
        self.rp.write_to_excel_simple_data(file_name=parsing_file_path, data=delta_t21,
                                        sheetname='DLT_Parse', chart_title='T2 - T1: DGW Send RQST - DGW Get RQST')

        parsing_file_path = f'{self.rp.log_dir}{TEST_TYPE}Parsing_T4_T3_{self.rp.get_time()}{self.rp.file_ext}'
        self.rp.write_to_excel_simple_data(file_name=parsing_file_path, data=delta_t43,
                                        sheetname='DLT_Parse', chart_title='T4 - T3: DGW Send RESP - DGW Get RESP')

        
if __name__ == '__main__':
    parser = DLTParser()
    parser.test_dlt_parser_script()
