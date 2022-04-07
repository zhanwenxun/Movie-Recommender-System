# Movie Recommender System

Duration: 2022.3 - 2022.4

Team:	Zhan Wenxun

​		 Xu Mingyang

​		 Liu Jiaqi

### Project File Structure

- Data File：

	- movie_info.xlsx	Movie Inforamtion Data

	- movie_poster.xlsx  Movie Poster Data
	- movies.xlsx        Movie Complete Information (Contains the above two data)
	- u.data             MovieLens 1M raw rating data
	- new_u.data         Added data for new user ratings
	- predict.data       predicted score recommended by the CFR



- Main File:
	- index.html         Recommender system front-end webpage file
	- main.py            Recommender system backend implementation file
	- create_file.py     merge movie_info.xlsx and movie_poster.xlsx to output movies.xlsx
	- utils.py           Define the tool functions that will be used in the project
	- cbr_recommend.py   Content Based Recommendation achieve function
	- cbr_evaluate.py    Content Based Recommendation evalution function
	- tagcloud.py        Generate current user's personalized Tag Cloud



- Generated File:
	- model              Model file that records the CFR recommendation algorithm
	- tagcloud.jpg       Personalized TagCloud image file of the current user

### Deploy And Run

1. Pull the project to the local, and open the project folder with VS Code
2. Open the terminal and run the 'pip install -r requirements.txt' command to install the python library required for the project to run (for VS Code, please configure the python core by yourself)
3. Run the 'uvicorn main:app --port 8887' command to run the project
4. Use the VS Code extension Live Sever or the Chrome extension to open the webpage

### Tips

- If the VS Code terminal reports an error pip or uvicorn is not a cmdlet command, it means that you have not configured the environment variables for the Python core used by VS Code, please restart the computer and run it again.
- If you are using Live Server in step 4, a second dialog box may appear and the page will be refreshed after scoring. This is because the project data file has stored the user's new score. Run through the project. The problem can be solved by modifying VS Code's settings.json with the following code:

VS Code: File -> Preferences -> Settings -> Search Live Server -> Edit in settings.json

```json
    "liveServer.settings.donotVerifyTags": true,
    "window.zoomLevel": -1,
    "terminal.integrated.environmentChangesRelaunch": false,
    "liveServer.settings.fullReload": true,
    "liveServer.settings.ignoreFiles": [
        ".vscode/**",
        "**/*.scss",
        "**/*.sass",
        "**/*.data",
        "**/*.xlsx",
        "**" // keep Live Server run whatever project files are edited
    ],
    "liveServer.settings.ChromeDebuggingAttachment": false
}
```