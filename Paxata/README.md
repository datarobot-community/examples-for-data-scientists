# Paxata Functions

Scope: The scope of this script is to provide you with ready python functions which you can use to interact with DataRobot Paxata.

### Table of contents

(1) Get Library Name and schema from LibraryID

(1a) Get Library Name and schema from LibraryID (and Version)

(2) Get all of the datasources from Paxata that are tagged with "tag"

(3) POST Library data from Paxata and load it into a JSON structure

(4) Get the Name of a Library from it's DatasetID

(5) Get all the Projects that have been described with a specific "description_tag"

(6) Post a file to the Paxata library (Paxata will guess how to parse it), return the new libraryId

(7) Check if a Project Name exists and return it's ID

(8) Delete a project based on ID (TEST THIS), not sure if i can access the content directly

(9) Post (Run) a Project

(10) Get the script of a projectId

(11) Replace values in a json file (useful for updating Paxata project scripts)

(12) Create a new (empty) Paxata project. Will return the projectId

(13) Update an existing Project with a new script file (this is not recommended)

(14) Get the script of a projectId

(15) Delete a Library item from it's ID

(16) Export/POST a libraryItem(Answerset) to a target

(17) Get DatasourceId and ConnectorId from Name of the Datasource

(18) Get the UserID From the REST API Token

(19) Check that an Export/POST to an external source has been completed (version 2.22)

(20) Check that an Export/POST to an external source has been completed (all versions)

(21) Get the datasetId of a Library from it's Name (a potentially very expensive call)

(22) Get all of the datasources (ordered_ for a tenant)

(23) Get All columns names for a library item

(25) Extract key values out of a json file def extract_values(obj, key)

(26) Get the Library items of a Project

### User Admin Functions

(24) Get ALl users on a tenant

(30) get_name_latest_version_and_schema_of_datasource

**Requirements:** Python 3.7 or higher;