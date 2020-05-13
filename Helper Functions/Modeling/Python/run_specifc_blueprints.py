#Author: Thodoris Petropoulos

def run_specific_blueprints(project_object, search_term, featurelist_id = None):
    """Runs all of the blueprints that match the search term use provides
        Input:
        - project_object <DataRobot Project> (Your DataRobot project)
        - search_term <string> (What to search for in the name of the Blueprint. e.g: "Gradient") 
        - featurelist_id <DataRobot Featurelist id> (Optional parameter to specify featurelist to use)
    """

    blueprints = project_object.get_blueprints()
    models_to_run = [blueprint for blueprint in blueprints if blueprint.model_type == search_term]
    for model in models_to_run:
        project_object.train(model, sample_pct = 80, featurelist_id=featurelist_id)

    while len(project_object.get_all_jobs()) > 0:
    time.sleep(1)
    pass
        
    