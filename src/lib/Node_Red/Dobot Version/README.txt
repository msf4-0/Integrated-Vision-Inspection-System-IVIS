Install Node.js https://nodejs.org/en/
Install node red dashboard https://flows.nodered.org/node/node-red-dashboard
Install node red table ui https://flows.nodered.org/node/node-red-node-ui-table
Open node red in command prompt by typing node-red 
Copy the link http://127.0.0.1:1880/ and paste into supported web browser 
Go to the hamburger menu on the top right corner and select import, then upload the js file Flow_Coord_Label_View into node-red 
Click deploy on the top right corner 
To bring up the UI enter http://127.0.0.1:1880/ui into the web broswer 
The python file Dobot_Version which will create the CSV file has to be run before anything is done in node-red 
The current location is set to save the CSV file into the desktop

Note* dobot_api file contains code necessary to interface with the dobot properly
This version of the code was made for the Dobot CR5 but easy steps can be taken to modify the files for dobot MG400 