library(jsonlite)
library(httr)
library(glue)
library(tidyverse)
library(doParallel)

# API request function ####
asa_api_request <- function(path = "https://app.americansocceranalysis.com/api/v1", 
                            league = "mls",
                            endpoint = "teams/xgoals",
                            parameters = NULL) {
  parameters_array <- c()
  
  if (length(parameters) > 0) {
    for (i in 1:length(parameters)) {
      tmp_name <- names(parameters[i])
      tmp_value <- parameters[[tmp_name]]
      
      if (all(!is.na(tmp_value)) & all(!is.null(tmp_value))) {
        if (length(tmp_value) > 1) {
          tmp_value <- gsub("\\s+", "%20", paste0(tmp_value, collapse = ","))
        } else {
          tmp_value <- gsub("\\s+", "%20", tmp_value)
        }
        
        parameters_array <- c(parameters_array, paste0(tmp_name, "=", tmp_value))
      }
    }
  }
  
  parameters_array <- ifelse(length(parameters_array) > 0,
                             paste0("?", paste0(parameters_array, collapse = "&")),
                             "")
  
  
  return(fromJSON(content(GET(glue("{path}/{league}/{endpoint}{parameters_array}")),
                          as = "text", encoding = "UTF-8")))
}

# # Example parameters for player data
# parameters_input <- list(
#   above_replacement = NULL,
#   general_position = NULL,
#   stage_name = NULL,
#   split_by_seasons = TRUE,
#   split_by_teams = NULL,
#   start_date = "2013-09-20",
#   end_date = "2013-09-27",
#   season_name = NULL,
#   team_id = NULL,
#   minimum_minutes = NULL,
#   gamestate_trunc = NULL,
#   zone = NULL)

# Aggregate daily player g+ data ####
if(!file.exists("Data/player_gplus_bygame.Rds")){
  cl <- makeCluster(8)
  registerDoParallel(cl)
  gplus_players <- foreach(Date = seq.Date(as.Date("2013-03-01"), as.Date(Sys.Date()), by = "days"),
                     .inorder = FALSE,
                     .combine = bind_rows,
                     .packages = c("httr", "jsonlite", "glue", "tidyr", "dplyr"))%dopar%{
                       
                       parameters_input <- list(
                         above_replacement = NULL,
                         general_position = NULL,
                         stage_name = NULL,
                         split_by_seasons = TRUE,
                         split_by_teams = NULL,
                         start_date = Date,
                         end_date = Date,
                         season_name = NULL,
                         team_id = NULL,
                         minimum_minutes = NULL,
                         gamestate_trunc = NULL,
                         zone = NULL) 
                       
                       df <- asa_api_request(path = "https://app.americansocceranalysis.com/api/v1", 
                                             league = "mls",
                                             endpoint = "players/goals-added",
                                             parameters = parameters_input)
                       
                       if(is.data.frame(df)){
                         return(unnest(df, cols = c(data)) %>% mutate(Date = Date) %>% select(-season_name))
                       } else{
                         return(data.frame())
                       }
                     }
  
  stopCluster(cl)
  rm(cl)
  gc()
  
  saveRDS(gplus_players, 
          "Data/player_gplus_bygame.Rds")
} else{
  gplus_players <- readRDS("Data/player_gplus_bygame.Rds")
}

# Aggregate daily goalkeeper g+ data ####
if(!file.exists("Data/keeper_gplus_bygame.Rds")){
  cl <- makeCluster(8)
  registerDoParallel(cl)
  gplus_keepers <- foreach(Date = seq.Date(as.Date("2013-03-01"), as.Date(Sys.Date()), by = "days"),
                     .inorder = FALSE,
                     .combine = bind_rows,
                     .packages = c("httr", "jsonlite", "glue", "tidyr", "dplyr"))%dopar%{
                       
                       parameters_input <- list(
                         above_replacement = NULL,
                         general_position = NULL,
                         stage_name = NULL,
                         split_by_seasons = TRUE,
                         split_by_teams = NULL,
                         start_date = Date,
                         end_date = Date,
                         season_name = NULL,
                         team_id = NULL,
                         minimum_minutes = NULL,
                         gamestate_trunc = NULL,
                         zone = NULL)
                       
                       df <- asa_api_request(path = "https://app.americansocceranalysis.com/api/v1",
                                             league = "mls",
                                             endpoint = "goalkeepers/goals-added",
                                             parameters = parameters_input)
                       
                       if(is.data.frame(df)){
                         return(unnest(df, cols = c(data)) %>% mutate(Date = Date) %>% select(-season_name))
                       } else{
                         return(data.frame())
                       }
                     }
  
  stopCluster(cl)
  rm(cl)
  gc()
  
  saveRDS(gplus_keepers,
          "Data/keeper_gplus_bygame.Rds")
} else{
  gplus_keepers <- readRDS("Data/keeper_gplus_bygame.Rds")
}

# Get/clean historical team xGoals results ####
if(!file.exists("Data/team_xg_bygame.Rds")){
  xg_teams <- asa_api_request(path = "https://app.americansocceranalysis.com/api/v1", 
                           league = "mls",
                           endpoint = "games/xgoals")
  
  xg_teams <- xg_teams %>%
    mutate(Date = as.Date(date_time_utc)) %>%
    select(game_id, Date, home_team_id, away_team_id, home_team_xgoals, away_team_xgoals)
  
  saveRDS(xg_teams,
          "Data/team_xg_bygame.Rds")
} else{
  xg_teams <- readRDS("Data/team_xg_bygame.Rds")
}

# Get primary dataset (actual team results) ####
if(!file.exists("Data/team_results_primary.Rds")){
  team_results_primary <- asa_api_request(path = "https://app.americansocceranalysis.com/api/v1", 
                                          league = "mls",
                                          endpoint = "games")
  
  team_results_primary <- team_results_primary %>%
    mutate(Date = as.Date(date_time_utc)) %>%
    select(game_id, Date, home_team_id, away_team_id, home_score, away_score)
  
  saveRDS(team_results_primary,
          "Data/team_results_primary.Rds")
} else{
  team_results_primary <- readRDS("Data/team_results_primary.Rds")
}

# Add team/player names to datasets ####
# Get team information
team_info <- asa_api_request(path = "https://app.americansocceranalysis.com/api/v1", 
                           league = "mls",
                           endpoint = "teams")

player_info <- asa_api_request(path = "https://app.americansocceranalysis.com/api/v1", 
                         league = "mls",
                         endpoint = "players")

# Join team/player info
gplus_individual <- gplus_players %>%
  left_join(player_info %>% select(player_id, player_name),
            by = "player_id") %>%
  left_join(team_info %>% select(team_id, team_abbr = team_abbreviation),
            by = "team_id") %>%
  bind_rows(gplus_keepers %>%
              left_join(player_info %>% select(player_id, player_name),
                        by = "player_id") %>%
              left_join(team_info %>% select(team_id, team_abbr = team_abbreviation),
                        by = "team_id"))

saveRDS(gplus_individual,
        "Data/individuals_gplus_bygame.Rds")

team_history <- gplus_individual %>%
  group_by(field = ifelse(general_position == "GK",
                          "GK",
                          "Field"),
           action_type,
           Date,
           team_id,
           team_abbr) %>%
  summarize(goals_added_above_avg = sum(goals_added_above_avg)) %>%
  pivot_wider(names_from = c(action_type, field),
              values_from = goals_added_above_avg,
              values_fill = 0) %>%
  ungroup() %>%
  mutate(offensive_gplus = Dribbling_Field + Passing_Field + Receiving_Field + Shooting_Field + Passing_GK) %>%
  left_join(xg_teams %>% select(team_id = home_team_id, Date, home_team_xgoals),
            by = c("Date", "team_id")) %>%
  left_join(xg_teams %>% select(team_id = away_team_id, Date, away_team_xgoals),
            by = c("Date", "team_id")) %>%
  mutate(team_xgoals = ifelse(is.na(home_team_xgoals),
                              away_team_xgoals,
                              home_team_xgoals)) %>%
  select(-c(home_team_xgoals, away_team_xgoals))

saveRDS(team_history,
        "Data/team_gplus_xg_bygame.Rds")


