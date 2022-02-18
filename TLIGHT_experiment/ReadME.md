## Experimental Setup 

The Experiment was conducted using <i class="icon-cog"></i> **[OpenPLC](https://www.openplcproject.com)** 
and <i class="icon-cog"></i> **[ScadaBr](https://sourceforge.net/projects/scadabr/)**.

The files in this directory involve the following:
```
- build: build file for the OpenPLC editor. It's responsible for 
experimental simulation in OpenPLC editor.

- plc.xml: contains XML file for sim interface

- structured_text_program: contains a structured text PLC program 
(traffic_journal.st) which represents the logic of the TLIGHT experiment
```

## Usage:
*Please refer to the official website of <i class="icon-cog"></i> **[OpenPLC](https://www.openplcproject.com)**
for further instructions on the installation of OpenPLC editor and 
runtime*

 - Upload the build file, plc.xml, and beremiz.xml into OpenPLC editor in
order to run a simulation


 - For actual replication of TLIGHT experiment, the program needs to be 
run in OpenPLC runtime by following the instructions below:

   1. Follow the instructions **[here](https://www.openplcproject.com/reference/basics/upload)**
   to set up the OpenPLC runtime. In this case, your computer will mimic
   the architecture of a PLC.

   2. Go to the programs section on the menu 

   3. Click on "Choose File", select "traffic_journal.st" file in the structured_text_program
   directory and then click on Upload program.

   4. Fill the information (as you prefer) in the pop-up window and click on 
   "Upload program" once you are done loading the .st file into the runtime
   5. Go to the dashboard of the runtime and wait till the .st file finishes
   compiling.
   6. Once the program finishes compiling, the status will change to "running"
   and the TLIGHT program will start running. 
   7. Click on the "Stop PLC" button if you wish to stop the program.



