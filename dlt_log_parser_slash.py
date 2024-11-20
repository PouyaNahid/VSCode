"""
DLT Parser
"""
import os
from subprocess import run

import pandas as pd
import re
import slash

from tests.System_Integration.Rigil_HW_Only.pt_service_handler import REMOTE_ECUS_LIST, ReportGenerator

TEST_TYPE = 'Physical'  # Physical | Functional
INTERNAL_TESTER_ECUS = [258, 264, 266]  # 102, 108, 10A: connected to internal terster
SEARCH_KEYWORDS = {
    'Physical': {'KW_DGW_GET_RQST': ['DoipUdsRouterInstanceId method_name= IndicateMessage call_id'],
                 # + 3712 264 which is 0xE80 0x108

                 'KW_DM_INDICATE': ['IndicateMessage corrId:'],  # S1
                 # + SA: Tester ADDR + TA: ECU ADDR like: SA: E80 TA: 119
                 # ['IndicateMessage corrId:', SA: E80 TA: 119]

                 'KW_DM_BACK_INDICATE': ['IndicateMessage corrId:', 'isFunc: false'],  # S2
                 # + TA: ECU ADDR like TA: 119
                 # ['IndicateMessage corrId:, isFunc: false, TA: 119]

                 'KW_DM_HANDLE': ['HandleMessage corrId:'],  # S3,
                 # + SA: Tester ADDR TA: ECU ADDR like SA: E80 TA: 119
                 # [HandleMessage corrId:, SA: E80 TA: 119]

                 'KW_DGW_SEND_RQST': ['frameSender'],  # S4 = T2
                 # + External Tester ADDR(hex) + ECU addr(hex) like: 0E 80 01 08 which is 0xE80, 0x108
                 # ['frameSender', 0E 80 01 08]  or
                 # + Internal Tester ADDR(hex) + ECU addr(hex) like: 0F 00 01 02 which is 0xF00, 0x102
                 # ['frameSender', 0F 00 01 02]

                 'KW_DGW_GET_RESP': ['DiagMsg(8001)'],
                 # + SA: ECU ADDR + TA: Tester ADDR like SA: 108 + TA: E80
                 # ['External DiagMsg(8001)', SA: 108, TA: E80]
                 # ['External DiagMsg(8001)', SA: 108, TA: F00]

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
                   # ['External DiagMsg(8001)', SA: 108, TA: E80]
                   # ['External DiagMsg(8001)', SA: 108, TA: F00]

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

DLT_LOG_FILE_NAME = 'DLT_Load3_4K.dlt'
IT_DISCRETE_HEX_ADDR = '0F 00'
ET_DISCRETE_HEX_ADDR = '0E 80'
FUNC_DISCRETE_HEX_ADDR = 'E4 00'

IT_HEX_ADDR = 'F00'
ET_HEX_ADDR = 'E80'
FUNC_HEX_ADDR = 'E400'


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


class DLTParser(slash.Test, ReportGenerator):
    """This class is defined for Parsing DLT log file and extracet timestamps for predefined DGW Entries."""
    dgw_recv_send_rqst_resp_entry_kws = SEARCH_KEYWORDS[TEST_TYPE]
    dlt = None
    matched_rows = None
    min_recorded_match_sample = 10_000_000

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

    # def extract_matched_timestamps(self) -> list:
    #     """
    #     Extract the time information from the matched rows.

    #     Returns a list of time values in the format [hh, mm, ss.sss] for each row in the match_rows DataFrame.
    #     The time information is extracted from the 'Time' column of each row, and the extracted values are
    #     added to the output list.

    #     Returns:
    #     - list: A list containing time values in the format [hh, mm, ss.sss].
    #     """
    #     time_hh_mm_ssdotms = []  # like [[5, 42, 16.676019]]  for 2024/07/08 5:42:16.676019
    #     for _, row in self.matched_rows.iterrows():
    #         time_hh_mm_ssdotms.append(row['Time'][11:].split(':'))

    # return time_hh_mm_ssdotms

    def extract_matched_timestamps(self) -> list:
        """
            Extract the time information from the matched rows.

            Returns a list of time values in the format [hh, mm, ss.sss] for each row in the match_rows DataFrame.
            The time information is extracted from the 'Time' column of each row, and the extracted values are
            added to the output list.

            Returns:
            - list: A list containing time values in the format [hh, mm, ss.sss].
        """
        timestamps_corrIds = []
        for _, row in self.matched_rows.iterrows():
            # time_hh_mm_ssdotms.append(row['Time'][11:].split(':'))
            timestamp = pd.to_datetime(row['Time'])  # Convert to datetime object
            corrId_match = re.search(r'corrId:\s*(\d+)', row['Payload'])
            if corrId_match:
                corrId = int(corrId_match.group(1))
                timestamps_corrIds.append((timestamp, corrId))  # Store both timestamp and corrId
        return timestamps_corrIds

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

    # def calculate_delta_time_of_two_entry(self, timestamp_ecu_sample: dict, entry_a: str, entry_b: str):
    #     """
    #         Calculates the time difference between two entries (timestamps) for each ECU.

    #         Parameters:
    #             timestamp_ecu_sample (dict): A dictionary containing timestamps for each ECU and sample.
    #                 Example: {'ECU1': {'entry_a': [timestamp1, timestamp2, ...], 'entry_b': [timestamp1, timestamp2, ...]},
    #                         'ECU2': {'entry_a': [timestamp1, timestamp2, ...], 'entry_b': [timestamp1, timestamp2, ...]},
    #                         ...}
    #             entry_a (str): The name of the first (start time) entry.
    #             entry_b (str): The name of the second (stop time) entry.

    #         Returns:
    #             dict: A dictionary containing the time differences for each ECU.
    #                 Example: {'Iteration': [1, 2, ...], 'ECU1': [diff1, diff2, ...], 'ECU2': [diff1, diff2, ...], ...}
    #     """
    #     delta = {'Iteration': []}
    #     longest_response_length = 0
    #     # timestamps_ecu_sample is somthing like:
    #     # {
    #     #  258: {'KW_DGW_GET_RQST': [['11', '28', '05.953127'], ['11', '28', '07.189936']],
    #     #        'KW_DGW_SEND_RQST': [['11', '28', '05.951632'], ['11', '28', '07.185723']]},
    #     #  260: {'KW_DGW_GET_RQST': [['11', '28', '05.953127'], ['11', '28', '07.189936']],
    #     #        'KW_DGW_SEND_RQST': [['11', '28', '05.951632'], ['11', '28', '07.185723']]}
    #     # }
    #     for ecu, entries in timestamp_ecu_sample.items():
    #         delta[ecu] = []
    #         # Determine the available number of sampele to participate in time_diff calculation
    #         entry_a_times = entries.get(entry_a, [])
    #         entry_b_times = entries.get(entry_b, [])
    #         ecu_total_sample = min(len(entry_a_times), len(entry_b_times), self.min_recorded_match_sample)
    #         print('hhhhhhhhhhhhhey len', ecu_total_sample, self.min_recorded_match_sample)

    #         # Determine the maximum length of samples between ECUs
    #         longest_response_length = max(ecu_total_sample, longest_response_length)

    #         for sample in range(ecu_total_sample):
    #             diff = calculate_time_difference_ms(entry_a_times[sample], entry_b_times[sample])
    #             delta[ecu].append(diff)

    #         # To add the average of each column at the latest row of that column
    #         if len(delta[ecu]):
    #             delta[ecu].append(round(sum(delta[ecu]) / len(delta[ecu]), 2))
    #         else:
    #             delta[ecu].append(0)

    #     for ecu, diff_samples in delta.items():
    #         if len(diff_samples) < longest_response_length:
    #             diff_samples.extend([0] * (longest_response_length - len(diff_samples)))

    #     # Add iteration column for excel report and chart and Average row to the end of the table
    #     delta['Iteration'] = list(range(1, longest_response_length + 1)) + ['Average']
    #     # Add Average row to the end of the table

    #     return delta

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

        parsing_file_path = f'{self.log_dir}{TEST_TYPE}Parsing_T4_T1_{self.get_time()}{self.file_ext}'
        self.write_to_excel_simple_data(file_name=parsing_file_path, data=delta_t41,
                                        sheetname='DLT_Parse', chart_title='T4 - T1: DGW Send RESP - DGW Get RQST')

        parsing_file_path = f'{self.log_dir}{TEST_TYPE}Parsing_T2_T1_{self.get_time()}{self.file_ext}'
        self.write_to_excel_simple_data(file_name=parsing_file_path, data=delta_t21,
                                        sheetname='DLT_Parse', chart_title='T2 - T1: DGW Send RQST - DGW Get RQST')

        parsing_file_path = f'{self.log_dir}{TEST_TYPE}Parsing_T4_T3_{self.get_time()}{self.file_ext}'
        self.write_to_excel_simple_data(file_name=parsing_file_path, data=delta_t43,
                                        sheetname='DLT_Parse', chart_title='T4 - T3: DGW Send RESP - DGW Get RESP')
