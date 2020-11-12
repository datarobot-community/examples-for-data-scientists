__author__ = 'Callum Finlayson'

import requests,json,re
from collections import OrderedDict

# (1) Get Library Name and schema from LibraryID
def get_name_and_schema_of_datasource(auth_token,paxata_url,libraryId,library_version):
    url_request = (paxata_url + "/rest/library/data/"+str(libraryId)+"/"+str(library_version))
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    if(my_response.ok):
        jdata_datasources = json.loads(my_response.content)
        library_name = jdata_datasources[0].get('name')
        library_schema_dict = jdata_datasources[0].get('schema')
    return library_name,library_schema_dict

# (1a) Get Library Name and schema from LibraryID (and Version)
def get_name_and_schema_of_datasource(auth_token,paxata_url,libraryId,library_version):
    url_request = (paxata_url + "/rest/library/data/"+str(libraryId)+"/"+str(library_version))
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    library_name = ""
    library_schema_dict = {}
    if(my_response.ok):
        jdata_datasources = json.loads(my_response.content)
        library_name = jdata_datasources.get('name')
        library_schema_dict = jdata_datasources.get('schema')
    return library_name,library_schema_dict


# (2) Get all of the datasources from Paxata that are tagged with "tag"
def get_tagged_library_items(auth_token,paxata_url,tag):
    tagged_datasets = []
    get_tags_request = (paxata_url + "/rest/library/tags")
    get_tags_response = requests.get(get_tags_request, auth=auth_token, verify=False)
    if (get_tags_response.ok):
        AllTagsDatasetsJson = json.loads(get_tags_response.content)
        i=0
        number_of_datasets = len(AllTagsDatasetsJson)
        while i < number_of_datasets:
            if (AllTagsDatasetsJson[i].get('name') == tag):
                tagged_datasets.append(AllTagsDatasetsJson[i].get('dataFileId'), AllTagsDatasetsJson[i].get('name'))
            i += 1
    else:
        print("bad request> " + get_tags_response.status_code)
    return tagged_datasets

# (3) POST Library data from Paxata and load it into a JSON structure
def get_paxata_library_data(auth_token,paxata_url,library_dataset_id):
    post_request = (paxata_url + "/rest/datasource/exports/local/" + library_dataset_id + "?format=json")
    post_response = requests.post(post_request,auth=auth_token)
    if (post_response.ok):
        JsonData = json.loads(post_response.content, object_pairs_hook=OrderedDict)
    return JsonData

# (4) Get the Name of a Library from it's DatasetID
def get_name_of_datasource(auth_token_source,paxata_url_source,libraryId):
    url_request = (paxata_url_source + "/rest/library/data/"+str(libraryId))
    my_response = requests.get(url_request,auth=auth_token_source , verify=False)
    if(my_response.ok):
        jDataDataSources = json.loads(my_response.content)
        libraryName = jDataDataSources[0].get('name')
    return libraryName

# (5) Get all the Projects that have been described with a specific "description_tag"
def get_all_project_information(auth_token_source, paxata_url_source, description_tag):
    Package_Tagged_Projects = []
    package_counter = 0
    max_num_of_projects = 0
    ProjectNames = []
    url_request = (paxata_url_source + "/rest/projects")
    my_response = requests.get(url_request,auth=auth_token_source , verify=False)
    if(my_response.ok):
        jDataProjectIds = json.loads(my_response.content)
        for item in jDataProjectIds:
            if description_tag == jDataProjectIds[package_counter].get('description'):
                ProjectNames.append(jDataProjectIds[package_counter].get('name'))
                Package_Tagged_Projects.append(jDataProjectIds[package_counter].get('projectId'))
                max_num_of_projects += 1
            package_counter += 1
    return Package_Tagged_Projects

# (6) Post a file to the Paxata library (Paxata will guess how to parse it), return the new libraryId
def post_file_to_paxata_library(auth_token_target,paxata_url_target, new_file_name):
    new_libraryId = ""
    if new_file_name is None:
        print("File doesn't exist??")
    else:
        ds = str(new_file_name)
        print("Uploading \"" + str(new_file_name) + "\" to Library")
        sourcetype = {'source': 'local'}
        files = {'data': open(ds, 'rb')}
        dataset_upload_response = ""
        try:
            dataset_upload_response = requests.post(paxata_url_target + "/rest/datasource/imports/local", data=sourcetype,
                                                  files=files, auth=auth_token_target)
        except:
            print("Connection error. Please validate the URL provided: " + paxata_url_target.url)
        if not (dataset_upload_response.ok):
            print("Couldn't upload the library data. Status Code = " + str(dataset_upload_response.status_code))
        else:
            jDataDataSources = json.loads(dataset_upload_response.content)
            new_libraryId = jDataDataSources.get('dataFileId')
    return new_libraryId

# (7) Check if a Project Name exists and return it's ID
def check_if_a_project_exists(auth_token,paxata_url,project_name):
    projectId = ""
    url_request = (paxata_url + "/rest/projects?name=" + project_name)
    my_response = requests.get(url_request,auth=auth_token , verify=False)
    if(my_response.ok):
        jdata_new_project_response = json.loads(my_response.content)
        if (not jdata_new_project_response):
            projectId = 0
        else:
            projectId = jdata_new_project_response[0]['projectId']
    else:
        my_response.raise_for_status()
    return projectId

# (8) Delete a project based on ID (TEST THIS), not sure if i can access the content directly
def delete_a_project_if_it_exists(auth_token,paxata_url,projectId):
    url_request = (paxata_url + "/rest/projects/" + str(projectId))
    my_response = requests.delete(url_request,auth=auth_token , verify=False)
    if(my_response.ok):
        print("Project \"", my_response.content.get('name'), "\" deleted.")
    else:
        my_response.raise_for_status()

# (9) Run a Project and publish the answerset to the library
def run_a_project(auth_token,paxata_url,projectId):
    post_request = (paxata_url + "/rest/project/publish?projectId=" + projectId + "&all=true")
    postResponse = requests.post(post_request, auth=auth_token, verify=False)
    if (postResponse.ok):
        print("Project Run - ", projectId)
    else:
        print("Something went wrong with POST call ", str(postResponse))
    # I need to investigate the below, sometimes postResponse.content is a dict, sometimes a list, hence the two below trys
    try:
        AnswersetId = json.loads(postResponse.content)[0].get('dataFileId')
    except(AttributeError):
        AnswersetId = json.loads(postResponse.content).get('dataFileId', 0)
    return AnswersetId

# (10) Get the script of a projectId
def get_project_script(auth_token,paxata_url,projectId):
    url_request = (paxata_url + "/rest/scripts?projectId=" + projectId)
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    if (my_response.ok):
        json_of_empty_project = json.loads(my_response.content)
    else:
        json_of_empty_project = 0
        my_response.raise_for_status()
    #the below return has an index of 0 to only return the latest version
    return json_of_empty_project[0]

# (11) Replace values in a json file (useful for updating Paxata project scripts)
def replace_json_values(json_file,oldvalue,newvalue):
    fileinput = json.dumps(json_file)
    newfileoutput = []
    newfile = ""
    newfile = re.sub(oldvalue, newvalue, fileinput.rstrip())
    newfileoutput = json.loads(newfile)
    return newfileoutput

# (12) Create a new (empty) Paxata project. Will return the projectId
def create_a_new_project(auth_token_target,paxata_url_target,Project_Name):
    projectId = ""
    url_request = (paxata_url_target + "/rest/projects?name=" + Project_Name)
    my_response = requests.post(url_request,auth=auth_token_target , verify=False)
    if(my_response.ok):
        print("Project \"", Project_Name ,"\" created.")
        jdata_new_project_response = json.loads(my_response.content)
        projectId = jdata_new_project_response['projectId']
    else:
        if my_response.status_code == 409:
            print("Project Already Exists")
        else:
            my_response.raise_for_status()
    return projectId

# (13) Update an existing Project with a new script file (this is not recommended)
def update_project_with_new_script(auth_token,paxata_url,final_updated_json_script,projectId,working_path):
    url_request = (paxata_url + "/rest/scripts?update=script&force=true&projectId=" + str(projectId))
    s = {'script': json.dumps(final_updated_json_script)}
    my_response = requests.put(url_request, data=s, auth=auth_token, verify=False)
    if (not my_response.ok):
        #if there is a problem in updating the project, it would indicate a problem with the script, so lets output it
        with open(working_path + '/invalid_script_dump.json', 'w') as f:
            json.dump(final_updated_json_script, f)
        my_response.raise_for_status()

# (14) Get an existing Project's script file
def get_new_project_script(auth_token,paxata_url,projectId):
    url_request = (paxata_url + "/rest/scripts?projectId=" + projectId + "&version=" + "0")
    myResponse = requests.get(url_request, auth=auth_token, verify=False)
    if (myResponse.ok):
        # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
        json_of_empty_project = json.loads(myResponse.content)
    else:
        json_of_empty_project = 0
        myResponse.raise_for_status()
    return(json_of_empty_project)

# (15) Delete Library Data from LibraryID
def delete_library_item(auth_token,paxata_url,libraryId):
    url_request = (paxata_url + "/rest/library/data/"+str(libraryId))
    my_response = requests.delete(url_request, auth=auth_token, verify=False)
    if(my_response.ok):
        jdata_datasources = json.loads(my_response.content, object_pairs_hook=OrderedDict)
        library_name = jdata_datasources.get('name')
    return library_name

# (16) Export/POST a libraryItem(Answerset) to a target
def post_paxata_library_data(auth_token,paxata_url,library_dataset_id,pax_datasourceId,pax_connectorId):
    post_request = (paxata_url + "/rest/datasource/exports/" + pax_datasourceId +"/" + library_dataset_id + "?format=json")
    post_response = requests.post(post_request,auth=auth_token)
    if (post_response.ok):
        JsonData = json.loads(post_response.content, object_pairs_hook=OrderedDict)
    return JsonData

# (17) Get DatasourceId and ConnectorId from Name of the Datasource
def get_datasource_id_from_name(auth_token,paxata_url,datasource_name):
    url_request = (paxata_url + "/rest/datasource/configs")
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    if(my_response.ok):
        jdata_datasources = json.loads(my_response.content, object_pairs_hook=OrderedDict)
        row_count = 0
        for row in jdata_datasources:
            if jdata_datasources[row_count].get('name') == datasource_name:
                pax_datasourceId = jdata_datasources[0].get('dataSourceId')
                pax_connectorId = jdata_datasources[0].get('connectorId')
            row_count +=1
    return pax_datasourceId,pax_connectorId

# (18) Get the UserID From the REST API Token
def get_user_from_token(auth_token, paxata_url, resttoken):
    url_request = (paxata_url + "/rest/users?authToken=true")
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    if (my_response.ok):
        jdata_datasources = json.loads(my_response.content, object_pairs_hook=OrderedDict)
        row_count = 0
        for row in jdata_datasources:
            if jdata_datasources[row_count].get('authToken') == resttoken:
                pax_name = jdata_datasources[row_count].get('name')
                pax_userId = jdata_datasources[row_count].get('userId')
            row_count += 1
    return pax_userId, pax_name

# (19) Check that an Export/POST to an external source has been completed (version 2.22)
def get_paxata_export_status(auth_token,pax_url,pax_exportId):
    get_request = (pax_url + "/rest/library/exports?exportId=" + pax_exportId)
    get_response = requests.get(get_request,auth=auth_token)
    if (get_response.ok):
        jdata_datasources = json.loads(get_response.content)
        print("Succesfully have the exportId status")
        exportIdStatus = jdata_datasources[0].get('exportId')
    else:
        print("Unsucessfully tried to get the exportId Status of exportId - ", pax_exportId)
    return exportIdStatus

# (20) Check that an Export/POST to an external source has been completed (all versions)
def get_paxata_export_status(auth_token,pax_url,pax_exportId):
    get_request = (pax_url + "/rest/library/exports?exportId=" + pax_exportId)
    get_response = requests.get(get_request,auth=auth_token, verify=False)
    if (get_response.ok):
        # In version 2.22 postResponse.content is a dict, prior to that it is a list which i need to manually iterate through, hence the two below trys
        print('.')
        jdata_datasources = json.loads(get_response.content)
        try:
            exportIdState = jdata_datasources.get('state')
            exporttimeStarted = jdata_datasources.get('timeStarted')
            exporttimeFinished = jdata_datasources.get('timeFinished')
        except(AttributeError):
            row_count = 0
            for row in jdata_datasources:
                if jdata_datasources[row_count].get('exportId') == pax_exportId:
                    exportIdState = jdata_datasources[row_count].get('state')
                    exporttimeStarted = jdata_datasources[row_count].get('timeStarted')
                    exporttimeFinished = jdata_datasources[row_count].get('timeFinished')
                row_count += 1
    else:
        print("Unsucessfully tried to get the exportId Status of exportId - ", pax_exportId)
    return(exportIdState,exporttimeStarted,exporttimeFinished)

# (21) Get the datasetId of a Library from it's Name
def get_id_of_datasource(auth_token,paxata_url,dataset_name):
    url_request = (paxata_url + "/library/data/")
    my_response = requests.get(url_request,auth=auth_token , verify=False)
    dataFileId = 0
    if(my_response.ok):
        jdata_datasources = json.loads(my_response.content)
        row_count = 0
        for row in jdata_datasources:
            if jdata_datasources[row_count].get('name') == dataset_name:
                dataFileId = jdata_datasources[row_count].get('dataFileId')
            row_count +=1
    return dataFileId


# (22) Get all of the datasources (ordered_ for a tenant)
def get_datasource_configs(authorization_token,paxata_url):
    url_request = (paxata_url + "/rest/datasource/configs")
    myResponse = requests.get(url_request, auth=authorization_token, verify=True)
    if (myResponse.ok):
        json_of_datasource_configs = json.loads(myResponse.content)
    else:
        json_of_datasource_configs = 0
        myResponse.raise_for_status()
    dict_of_datasources = {}
    for item in json_of_datasource_configs:
#        dict_of_datasources[item.get('connectorId')] = item.get('name')
        dict_of_datasources[item.get('name')] = item.get('dataSourceId')

    dict_of_datasources['0'] = ' - No Connector - Data already exists in Paxata - '
    #returning a sorted dictionary
    return(OrderedDict(sorted(dict_of_datasources.items(), key=lambda kv:(kv[0].lower(),kv[1]))))

# (23) Get All columns names for a library item
def get_library_item_metadata(authorization_token,paxata_url,dataFileID):
    url_request = (paxata_url + "/rest/library/data/"+ dataFileID)
    my_response = requests.get(url_request, auth=authorization_token, verify=False)
    if(my_response.ok):
        json_of_library_items = json.loads(my_response.content)
    else:
        json_of_library_items = 0
        my_response.raise_for_status()
    list_of_library_columns = []
    if json_of_library_items[0]['schema']:
        for item in json_of_library_items[0]['schema']:
            if item.get('type') == "String":
                list_of_library_columns.append((str(item.get('name'))))
    return list_of_library_columns

# (24) Get ALl users on a tenant
def get_users_on_tenant(authorization_token,paxata_url):
    url_request = (paxata_url + "/rest/users")
    my_response = requests.get(url_request, auth=authorization_token, verify=False)
    if(my_response.ok):
        json_of_library_items = json.loads(my_response.content)
    else:
        json_of_library_items = 0
        my_response.raise_for_status()
    list_of_library_columns = []
    if json_of_library_items[0]['schema']:
        for item in json_of_library_items[0]['schema']:
            if item.get('type') == "String":
                list_of_library_columns.append((str(item.get('name'))))
    return list_of_library_columns

# (25) Extract key values out of a json file
def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

# (26) Get the Library items of a Project
def get_libraryIds_of_lenses_exported_by_project(auth_token,paxata_url,projectId):
    url_request = (paxata_url + "/project/publish?projectId="+ projectId)
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    if(my_response.ok):
        json_of_library_items = json.loads(my_response.content)
    else:
        json_of_library_items = 0
        my_response.raise_for_status()
    return json_of_library_items


# (27) Get the information of where a =Library item was exported (ie which external system)
def get_info_of_exported_library_items(auth_token,paxata_url,libraryId):
    url_request = (paxata_url + "/library/exports")
    my_response = requests.get(url_request, auth=auth_token, verify=False)
    json_of_library_items = []
    if(my_response.ok):
        library_export_info = json.loads(my_response.content)
        for item in library_export_info:
            if item.get('dataFileId') == libraryId:
                i+=1
                print(item.get('destination'))
                print("Data Export - " ,str(i) + " for libraryId(" ,libraryId + ")")
                print("Filename = " , item.get('destination').get('name'))
                print("Path = " , item.get('destination').get('itemPath'))
                print("ConnectorID = " , item.get('destination').get('connectorId') + "\n")
                json_of_library_items.append(str(item.get('destination')))
    else:
        json_of_library_items = 0
        my_response.raise_for_status()
    return json_of_library_items


# (28) Update an existing Project's script file
def update_project_with_new_script(auth_token,paxata_url,updated_json_script,projectId):
    url_request = (paxata_url + "/rest/scripts?update=script&force=true&projectId=" + str(projectId))
    s = {'script': json.dumps(updated_json_script)}
    myResponse = requests.put(url_request, data=s, auth=auth_token)
    result = False
    print(myResponse)
    if (myResponse.ok):
        # json_of_existing_project = json.loads(myResponse.content)
        result = True
    else:
        #if there is a problem in updating the project, it would indicate a problem with the script, so lets output it
        print(myResponse.content)
        result = False
    return result

# (29) Update an existing Project's Datasource file
def update_project_with_new_dataset(auth_token,paxata_url,updated_json_script,projectId):
    url_request = (paxata_url + "/rest/scripts?update=datasets&force=true&projectId=" + str(projectId))
    s = {'script': json.dumps(updated_json_script)}
    myResponse = requests.put(url_request, data=s, auth=auth_token)
    result = False
    print(myResponse)
    if (myResponse.ok):
        # json_of_existing_project = json.loads(myResponse.content)
        result = True
    else:
        #if there is a problem in updating the project, it would indicate a problem with the script, so lets output it
        print(myResponse.content)
        result = False
    return result

# (30) get_name_latest_version_and_schema_of_datasource
def get_name_latest_version_and_schema_of_datasource(auth_token,paxata_url,libraryId):
    url_request = (paxata_url + "/library/data/"+str(libraryId))
    my_response = requests.get(url_request, auth=auth_token, verify=True)
    library_name = ""
    library_schema_dict = ""
    if(my_response.ok):
        jdata_datasources = json.loads(my_response.content)
        library_name = jdata_datasources[0].get('name')
        library_version = jdata_datasources[0].get('version')
        library_schema_dict = jdata_datasources[0].get('schema')
    return library_name,library_version,library_schema_dict
