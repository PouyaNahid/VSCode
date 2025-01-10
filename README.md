# VSCode
VSCode settings and points

## VSCode Setup For New Projects:
https://www.youtube.com/watch?v=mpk4Q5feWaw

### Folder Location
I suggest setting up a Repositories folder on your drive to keep all your code projects organized; you can even divide them into Personal Repositories and Work Repositories for better clarity.

1. In Projects Repository Create Local Folder for the new Project

2. Open the Folder using VSCode and Trust it by VSCode
### Opening a Folder
To open a folder in VS Code using the Explorer, simply navigate to "File" on the menu bar and select "Open Folder," then choose the directory you wish to access.

4. Create and save workspace for that folder (project)

### Saving the Workspace
To save your workspace file along with the folder in Visual Studio Code, first ensure all files are saved, then go to the File menu, select Save Workspace As, choose your desired location, give your workspace a name, and click Save.

4. In Github, create a Project files and folders and make it as Default Template for upcoming Projects
5. Make a new repository based on this template (name it based on your project name)
6. Clone the new repository to your local machine
7. Create virtual Environment for the project using VSCode.

### Virtual Environments
Open Command Palette: Use the shortcut to open the Command Palette in VS Code.
Select Environment Command: Type and select Python: Select Interpreter to choose an existing interpreter or create a new virtual environment.

Create New Environment:
Venv: Click on Venv → select Python version → requirements.txt (optional)
Check Environment: After the creation, double check to make sure the right environment is selected. You can adjust using Python: 
#### Select Interpreter
8. Active and use jupyter notebook interactive mode. The benefit of this mode is we can run a selected part of the code in a file and even add more code and just run the rest, which is able to use the memory data to read pre-run variables values. Instead of having print and adding to our code to see the results.

Interactive Python: Search for Jupyter Interactive Window → Enable (When pressing shift + enter, send selection to Jupyter interactive window as opposed to the Python terminal)
Regarding changing the root folder for Jupyter notebooks, you can modify your settings in VS Code by including the JSON snippet below:

``` json
"settings": {
                "jupyter.notebookFileRoot": "${workspaceFolder}/app",
        }
```

