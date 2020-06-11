#Author: Katy Chow Haynie

#Make sure you are connected to DataRobot Client.


#This function will help you create a copy of a TS project using the same exact settings. Manipulate as you see fit.

import datarobot as dr

def clone_ts_project(pid):
    """This function will copy a DataRobot TS project with the same settings
        Input:
         - pid <str> the id of the project you want to copy
         
        Manipulate this function as you see fit in case you dont want a complete 1:1 copy.
    """
    p = dr.Project.get(pid)
    c_p = p.clone_project('Clone of {}'.format(p.project_name))
    c_pid = c_p.id
    
    #Get datetimePartitioning data
    #This will include calendar, backtesting and known-in-advance features.
    dtp = dr.DatetimePartitioning.get(pid)
    dtp_spec_for_clone = dtp.to_specification()
    
    #Fix the datetime_partition_column which will have an ' (actual)' string appended to it.
    dtp_spec_for_clone.datetime_partition_column = dtp_spec_for_clone.datetime_partition_column.replace(' (actual)','')
    
    ##Place changes below##
    #Manipulate dtp_spec_for_clone as you see fit (you can directly change its attribites)
    
    ##
    
    c_p.set_target(target = 'Sales',
                       partitioning_method = dtp_spec_for_clone,
                       mode = 'auto',
                       worker_count = -1
                  )
    
##Usage##
#clone_ts_project('YOUR_PROJECT_ID')