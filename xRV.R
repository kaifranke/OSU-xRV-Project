library(dplyr)
library(xgboost)
library(caret)
library(MLmetrics)
library(ggplot2)
library(readr)
library(stringr)

data2022 <- read_csv("pitches2022.csv")
data1821 <- read_csv("2018-2021data.csv")

data = rbind(data1821, data2022)
data2 = rbind(data1821, data2022)

data = data %>%
  arrange(game_date, home_team, away_team, inning, desc(inning_topbot), at_bat_number, pitch_number) %>%
  select(pitch_type, release_speed, release_pos_x, release_pos_z, release_pos_y, pfx_x, pfx_z, plate_x, plate_z, delta_run_exp)

data2 = data2 %>%
  arrange(game_date, home_team, away_team, inning, desc(inning_topbot), at_bat_number, pitch_number) %>%
  mutate(pitchTeam = ifelse(inning_topbot == "Top", home_team, away_team)) %>%
  select(player_name, game_date, game_year, pitchTeam, home_team, away_team, balls, strikes, inning, inning_topbot, outs_when_up, description, des, events, pitch_type, release_speed, release_pos_x, release_pos_z, release_pos_y, pfx_x, pfx_z, plate_x, plate_z, estimated_woba_using_speedangle, delta_run_exp)


fast = data %>%
  filter(pitch_type %in% c("FF", "SI"),
         !is.na(release_speed),
         !is.na(release_pos_x),
         !is.na(release_pos_z),
         !is.na(release_pos_y),
         !is.na(pfx_x),
         !is.na(pfx_z),
         !is.na(plate_x),
         !is.na(plate_z),
         !is.na(delta_run_exp))
Fpart = createDataPartition(fast$delta_run_exp, p = .75, list = FALSE, times = 1)
trainF = fast[Fpart,]
testF = fast[-Fpart,]
fastD = xgb.DMatrix(data = as.matrix(trainF[2:9]), label = trainF$delta_run_exp)
togF = rbind(trainF, testF)



mvt = data %>%
  filter(pitch_type %in% c("SL", "KC", "CU", "FC"),
         !is.na(release_speed),
         !is.na(release_pos_x),
         !is.na(release_pos_z),
         !is.na(release_pos_y),
         !is.na(pfx_x),
         !is.na(pfx_z),
         !is.na(plate_x),
         !is.na(plate_z),
         !is.na(delta_run_exp))
Mpart = createDataPartition(mvt$delta_run_exp, p = .75, list = FALSE, times = 1)
trainM = mvt[Mpart,]
testM = mvt[-Mpart,]
mvtD = xgb.DMatrix(data = as.matrix(trainM[2:9]), label = trainM$delta_run_exp)
togM = rbind(trainM, testM)


off = data %>%
  filter(pitch_type %in% c("CH", "FS"),
         !is.na(release_speed),
         !is.na(release_pos_x),
         !is.na(release_pos_z),
         !is.na(release_pos_y),
         !is.na(pfx_x),
         !is.na(pfx_z),
         !is.na(plate_x),
         !is.na(plate_z),
         !is.na(delta_run_exp))
Opart = createDataPartition(off$delta_run_exp, p = .75, list = FALSE, times = 1)
trainO = off[Opart,]
testO = off[-Opart,]
offD = xgb.DMatrix(data = as.matrix(trainO[2:9]), label = trainO$delta_run_exp)
togO = rbind(trainO, testO)



#### Offspeed Model ####


eta.list = c(0.05, 0.1)
max.depth.list = c(2,3)
subsample.list = c(0.5, 1)
colsamptree.list = c(0.5, 0.75, 1)
nround.list = c(350,400,450)


modNum = 0
result = matrix(NA, nrow = 2*2*1*3*3, ncol = 6)

set.seed(123)

for (i in 1:length(eta.list)) {
  for (j in 1:length(max.depth.list)) {
    for (k in 1:length(subsample.list)) {
      for (l in 1:length(colsamptree.list)) {
        for (m in 1:length(nround.list)) {
          modNum = modNum + 1
          
          xgb_model = xgb.train(data = offD,
                                eta = eta.list[i],
                                max_depth = max.depth.list[j],
                                subsample = subsample.list[k],
                                colsample_bytree = colsamptree.list[l],
                                nrounds = nround.list[m],
                                objective = 'reg:squarederror')
          
          testO$pred = predict(xgb_model, newdata = as.matrix(testO[2:9]))
          
          rmse = RMSE(testO$pred, testO$delta_run_exp)
          
          result[modNum, 1] = eta.list[i]
          result[modNum, 2] = max.depth.list[j]
          result[modNum, 3] = subsample.list[k]
          result[modNum, 4] = colsamptree.list[l]
          result[modNum, 5] = nround.list[m]
          result[modNum, 6] = rmse
        }
      }
    }
  }
}

colnames(result) = c("eta", "max depth", "subsample", "colsamptree", "nround", "rmse")

eta = 0.05
max.depth.list = c(3,4,5)
subsample = 0.5
colsamptree.list = c(0.75, 1)
nround.list = c(300,350,400,450)
modNum = 0
result2 = matrix(NA, nrow = 1*3*1*2*4, ncol = 6)

for (i in 1:length(max.depth.list)) {
  for (j in 1:length(colsamptree.list)) {
    for (k in 1:length(nround.list)) {
      modNum = modNum + 1
      
      xgb_model = xgb.train(data = offD,
                            eta = eta,
                            max_depth = max.depth.list[i],
                            subsample = subsample,
                            colsample_bytree = colsamptree.list[j],
                            nrounds = nround.list[k],
                            objective = 'reg:squarederror')
      
      testO$pred = predict(xgb_model, newdata = as.matrix(testO[2:9]))
      
      rmse = RMSE(testO$pred, testO$delta_run_exp)
      
      result2[modNum, 1] = eta
      result2[modNum, 2] = max.depth.list[i]
      result2[modNum, 3] = subsample
      result2[modNum, 4] = colsamptree.list[j]
      result2[modNum, 5] = nround.list[k]
      result2[modNum, 6] = rmse
    }
  }
}

result2


xgb_model = xgb.train(data = offD,
                      eta = 0.05,
                      max_depth = 4,
                      subsample = 0.5,
                      colsample_bytree = 0.75,
                      nrounds = 350,
                      objective = 'reg:squarederror')

testO$pred = predict(xgb_model, newdata = as.matrix(testO[2:9]))
RMSE(testO$pred, testO$delta_run_exp)
sd(testO$delta_run_exp)

xgb.importance(colnames(off[2:9]), model = xgb_model)


togO = rbind(trainO, testO[-11])

togO$pred = predict(xgb_model, newdata = as.matrix(togO[2:9]))
RMSE(togO$pred, togO$delta_run_exp)
sd(togO$delta_run_exp)



#### Fastball Model ####

set.seed(123)

eta = 0.05
max.depth.list = c(3,4,5)
subsample = 0.5
colsamptree.list = c(0.75, 1)
nround.list = c(300,350,400,450)
modNum = 0
result2 = matrix(NA, nrow = 1*3*1*2*4, ncol = 6)
togF = rbind(trainF, testF)

for (i in 1:length(max.depth.list)) {
  for (j in 1:length(colsamptree.list)) {
    for (k in 1:length(nround.list)) {
      modNum = modNum + 1
      
      xgb_model = xgb.train(data = fastD,
                            eta = eta,
                            max_depth = max.depth.list[i],
                            subsample = subsample,
                            colsample_bytree = colsamptree.list[j],
                            nrounds = nround.list[k],
                            objective = 'reg:squarederror')
      
      testF$pred = predict(xgb_model, newdata = as.matrix(testF[2:9]))
      
      rmse = RMSE(testF$pred, testF$delta_run_exp)
      
      result2[modNum, 1] = eta
      result2[modNum, 2] = max.depth.list[i]
      result2[modNum, 3] = subsample
      result2[modNum, 4] = colsamptree.list[j]
      result2[modNum, 5] = nround.list[k]
      result2[modNum, 6] = rmse
    }
  }
}

result2


xgb_modelF = xgb.train(data = fastD,
                      eta = 0.05,
                      max_depth = 5,
                      subsample = 0.5,
                      colsample_bytree = 0.75,
                      nrounds = 250,
                      objective = 'reg:squarederror')

testF$pred = predict(xgb_modelF, newdata = as.matrix(testF[2:9]))
RMSE(testF$pred, testF$delta_run_exp)
sd(testF$delta_run_exp)

xgb.importance(colnames(fast[2:9]), model = xgb_modelF)


togF$pred = predict(xgb_modelF, newdata = as.matrix(togF[2:9]))
RMSE(togF$pred, togF$delta_run_exp)
sd(togF$delta_run_exp)



#### Movement Model ####

set.seed(123)

eta = 0.05
max.depth.list = c(3,4,5)
subsample = 0.5
colsamptree.list = c(0.75, 1)
nround.list = c(300,350)
modNum = 0
result3 = matrix(NA, nrow = 1*3*1*2*2, ncol = 6)

for (i in 1:length(max.depth.list)) {
  for (j in 1:length(colsamptree.list)) {
    for (k in 1:length(nround.list)) {
      modNum = modNum + 1
      
      xgb_model = xgb.train(data = mvtD,
                            eta = eta,
                            max_depth = max.depth.list[i],
                            subsample = subsample,
                            colsample_bytree = colsamptree.list[j],
                            nrounds = nround.list[k],
                            objective = 'reg:squarederror')
      
      testM$pred = predict(xgb_model, newdata = as.matrix(testM[2:9]))
      
      rmse = RMSE(testM$pred, testM$delta_run_exp)
      
      result3[modNum, 1] = eta
      result3[modNum, 2] = max.depth.list[i]
      result3[modNum, 3] = subsample
      result3[modNum, 4] = colsamptree.list[j]
      result3[modNum, 5] = nround.list[k]
      result3[modNum, 6] = rmse
    }
  }
}

result3


xgb_modelM = xgb.train(data = mvtD,
                      eta = 0.05,
                      max_depth = 5,
                      subsample = 0.5,
                      colsample_bytree = 0.75,
                      nrounds = 250,
                      objective = 'reg:squarederror')


testM$pred = predict(xgb_modelM, newdata = as.matrix(testM[2:9]))
RMSE(testM$pred, testM$delta_run_exp)
sd(testM$delta_run_exp)

xgb.importance(colnames(mvt[2:9]), model = xgb_modelM)



togM$pred = predict(xgb_modelM, newdata = as.matrix(togM[2:9]))
RMSE(togM$pred, togM$delta_run_exp)
sd(togM$delta_run_exp)


saveRDS(xgb_modelM, file = "xRV_Mvt_Model.rda")
saveRDS(xgb_modelF, file = "xRV_Fst_Model.rda")
saveRDS(xgb_model, file = "xRV_Off_Model.rda")

#### Together ####


tog = rbind(togF, togM, togO)

dataFin = inner_join(data2, tog, by = c("pitch_type", "release_speed", "release_pos_x", "release_pos_z", "release_pos_y",
                             "pfx_x", "pfx_z", "plate_x", "plate_z", "delta_run_exp"))

dataFin = dataFin %>%
  mutate(`xRV+` = (100 * pred) / mean(pred)) %>%
  mutate(hitTeam = ifelse(inning_topbot == "Top", away_team, home_team))

players = dataFin %>%
  group_by(player_name, pitch_type, game_year) %>%
  dplyr::summarize(n = n(),
            RV = sum(delta_run_exp),       
            xRV = sum(pred),
            xRV.100 = (xRV / n) * 100)

batters = dataFin %>%
  mutate(batter = word(des, 1,2, sep=" ")) %>%
  group_by(batter, game_year) %>%
  summarize(n = n(),
            xRV = sum(pred),
            RV = sum(delta_run_exp),
            xwOBA = mean(estimated_woba_using_speedangle, na.rm = T))

batters = batters %>%
  filter(batter != "Umpire reviewed") %>%
  mutate(RVd = RV - xRV,
         RVdP = RVd / n) %>%
  arrange(batter, game_year) %>%
  mutate(nextxwOBA = ifelse(lead(batter) == batter, lead(xwOBA), NA),
         nextRVd = ifelse(lead(batter) == batter, lead(RVd), NA)) %>%
  filter(n >= 800)

ggplot(batters, aes(RVd, nextxwOBA)) +
  geom_point()

cor(filter(batters, !is.na(nextxwOBA))$RVd, filter(batters, !is.na(nextxwOBA))$nextxwOBA)^2
cor(filter(batters, !is.na(nextxwOBA))$xwOBA, filter(batters, !is.na(nextxwOBA))$nextxwOBA)^2
cor(filter(batters, !is.na(nextxwOBA))$RVd, filter(batters, !is.na(nextxwOBA))$nextRVd)^2


x <- c(-.7,.7,.7,-.7,-.7)
z <- c(1.6,1.6,3.5,3.5,1.6)
sz <- data_frame(x,z)

ggplot(sample_n(dataFin, 90000), aes(plate_x, plate_z, z = pred)) +
  stat_summary_hex() +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue") +
  coord_fixed() +
  ylim(-2.5, 6) +
  labs(title = "Predicted Run Value by Location", x = "Horizontal Location",
       y = "Vertical Location", fill = "xRV") +
  theme_bw()


dataFin = dataFin %>%
  mutate(Pdivision = ifelse(pitchTeam %in% c("MIN", "CLE", "CWS", "KC", "DET"), "ALC", NA),
         Pdivision = ifelse(pitchTeam %in% c("NYY", "BOS", "TOR", "TB", "BAL"), "ALE", Pdivision),
         Pdivision = ifelse(pitchTeam %in% c("SEA", "HOU", "LAA", "OAK", "TEX"), "ALW", Pdivision),
         Pdivision = ifelse(pitchTeam %in% c("CHC", "CIN", "MIL", "PIT", "STL"), "NLC", Pdivision),
         Pdivision = ifelse(pitchTeam %in% c("NYM", "WSH", "ATL", "MIA", "PHI"), "NLE", Pdivision),
         Pdivision = ifelse(pitchTeam %in% c("AZ", "SF", "ARI", "LAD", "SD", "COL"), "NLW", Pdivision))

dataFin = dataFin %>%
  mutate(Hdivision = ifelse(hitTeam %in% c("MIN", "CLE", "CWS", "KC", "DET"), "ALC", NA),
         Hdivision = ifelse(hitTeam %in% c("NYY", "BOS", "TOR", "TB", "BAL"), "ALE", Hdivision),
         Hdivision = ifelse(hitTeam %in% c("SEA", "HOU", "LAA", "OAK", "TEX"), "ALW", Hdivision),
         Hdivision = ifelse(hitTeam %in% c("CHC", "CIN", "MIL", "PIT", "STL"), "NLC", Hdivision),
         Hdivision = ifelse(hitTeam %in% c("NYM", "WSH", "ATL", "MIA", "PHI"), "NLE", Hdivision),
         Hdivision = ifelse(hitTeam %in% c("AZ", "SF", "ARI", "LAD", "SD", "COL"), "NLW", Hdivision),
         `xwOBA+` = (estimated_woba_using_speedangle - min(estimated_woba_using_speedangle, na.rm = T)) / (max(estimated_woba_using_speedangle, na.rm = T) - min(estimated_woba_using_speedangle, na.rm = T)),
         `xRV.100+` = (pred - min(pred)) / (max(pred) - min(pred)))


HdivSt = dataFin %>%
  group_by(Hdivision, game_year) %>%
  summarise(n = n(), 
            RV = sum(delta_run_exp),
            xRV = sum(pred),            
            RV.100 = (RV / n) * 100,
            xRV.100 = (xRV / n) * 100,
            xwOBA = mean(estimated_woba_using_speedangle, na.rm = T),
            soH = RV.100 - xRV.100) %>%
  arrange(desc(soH))


PdivSt = dataFin %>%
  group_by(Pdivision, game_year) %>%
  summarise(n = n(), 
            xRV = sum(pred),
            xRV.100 = (xRV / n) * 100) %>%
  arrange(xRV.100)


divs = inner_join(PdivSt, HdivSt, by = c("Pdivision" = "Hdivision", "game_year"))

divs = divs %>%
  mutate(scaleH = ((soH - (-.462)) / ((.753) - (-.462))),
         scaleP = ((-xRV.100.x) - (-.476)) / ((.188) - (-.476)) - 0.433,
         pwr = (scaleH + scaleP) / 2) %>%
  arrange(desc(pwr))




plfRec <- read_csv("Baseball Research/xRV/playoff records.csv")


divs2 = inner_join(divs, plfRec, by = c("Pdivision" = "DIV", "game_year" = "Year"))


divs2 = divs2 %>%
  mutate(Wp = (W) / (W + L))

ggplot(divs2, aes(pwr, Wp)) +
  geom_point()
cor(divs2$pwr, divs2$Wp)^2

divsSum = divs2 %>%
  group_by(Pdivision) %>%
  summarise(pwr = mean(pwr),
            Wp = mean(Wp))

ggplot(divsSum, aes(pwr, Wp, label = Pdivision)) +
  geom_point() +
  geom_text(nudge_x = 0.015) +
  theme_bw() +
  stat_smooth(method = "lm", se = F) +
  labs(title = "Division Power vs Playoff Winning Percentage", x = "Division Power", y = "Playoff Winning Percentage",
       subtitle = "Winning percentage only counts vs opposing divisions") +
  geom_text(x = 0.25, y = 0.55, label = paste("R^2:", round(cor(divsSum$pwr, divsSum$Wp)^2, 3)))
cor(divsSum$pwr, divsSum$Wp)^2


ggplot(filter(divs, game_year == 2022), aes(reorder(Pdivision, xRV.x), xRV.x)) +
  geom_col(aes(fill = Pdivision)) +
  geom_hline(yintercept = 0) +
  theme_bw() +
  labs(title = "Divisions by Expected Run Value in 2022", x = "Division", y = "Expected Pitching Run Value",
       fill = "Division")

ggplot(filter(divs, game_year == 2022), aes(reorder(Pdivision, soH), soH)) +
  geom_col(aes(fill = Pdivision)) +
  geom_hline(yintercept = 0) +
  theme_bw() +
  labs(title = "Divisions by RV Above Expected in 2022", x = "Division", y = "Runs Above Expected",
       fill = "Division")

ggplot(filter(divs, game_year == 2022), aes(reorder(Pdivision, pwr), pwr)) +
  geom_col(aes(fill = Pdivision)) +
  geom_hline(yintercept = 0) +
  theme_bw() +
  labs(title = "Divisions by Power in 2022", x = "Division", y = "Power",
       fill = "Division") +
  theme(legend.position = "none")


HteamSt = dataFin %>%
  group_by(hitTeam, game_year) %>%
  summarise(n = n(), 
            RV = sum(delta_run_exp),
            xRV = sum(pred),            
            RV.100 = (RV / n) * 100,
            xRV.100 = (xRV / n) * 100,
            xwOBA = mean(estimated_woba_using_speedangle, na.rm = T),
            soH = RV.100 - xRV.100) %>%
  arrange(desc(soH))


PteamSt = dataFin %>%
  group_by(pitchTeam, game_year) %>%
  summarise(n = n(), 
            xRV = sum(pred),
            xRV.100 = (xRV / n) * 100) %>%
  arrange(xRV.100)


teams = inner_join(PteamSt, HteamSt, by = c("pitchTeam" = "hitTeam", "game_year"))

teams = teams %>%
  mutate(scaleH = ((soH - (-.881)) / ((.753) - (-.881))),
         scaleP = ((-xRV.100.x) - (-.476)) / ((.380) - (-.476)) - 0.11,
         pwr = (scaleH + scaleP) / 2) %>%
  arrange(desc(pwr))


teams2022 = teams %>%
  mutate(scaleH = ((soH - (-.881)) / ((.753) - (-.881))),
         scaleP = ((-xRV.100.x) - (-.476)) / ((.380) - (-.476)) - 0.11,
         pwr = (scaleH + scaleP) / 2) %>%
  arrange(desc(pwr)) %>%
  filter(game_year == 2022)

teams %>%
  filter(game_year == 2022) %>%
  select(pitchTeam, pwr) %>%
  head(10)

head5 = head(teams2022, 5)

ggplot(head(teams2022, 7), aes(reorder(pitchTeam, -pwr), pwr, fill = pitchTeam)) +
  geom_bar(stat = "identity", color = "black") +
  geom_mlb_logos(aes(team_abbr = pitchTeam, width = 0.1)) +
  scale_fill_manual(values = c("red", "dark orange", "dodger blue", "royal blue", "navy", "light blue", "red")) +
                                       labs(x = "Team", y = "Power", title = "Top 7 Teams by their Power in 2022") +
  theme_bw() +
  theme(legend.position = "none",
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x = element_blank()) +
  ylim(0, 1.025)




ggplot(head(arrange(teams2022, desc(scaleH)), 7), aes(reorder(pitchTeam, -scaleH), scaleH, fill = pitchTeam)) +
  geom_bar(stat = "identity", color = "black") +
  geom_mlb_logos(aes(team_abbr = pitchTeam, width = 0.1)) +
  scale_fill_manual(values = c("red", "dark orange", "dodger blue", "royal blue", "navy", "red", "red")) +
  labs(x = "Team", y = "Hitting Power", title = "Top 7 Teams by their Hitting Power in 2022") +
  theme_bw() +
  theme(legend.position = "none",
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x = element_blank()) +
  ylim(0, 1.025)


ggplot(head(arrange(teams2022, desc(scaleP)), 7), aes(reorder(pitchTeam, -scaleP), scaleP, fill = pitchTeam)) +
  geom_bar(stat = "identity", color = "black") +
  geom_mlb_logos(aes(team_abbr = pitchTeam, width = 0.1)) +
  scale_fill_manual(values = c("dark orange", "dodger blue", "black", "navy", "navy", "orange", "light blue")) +
  labs(x = "Team", y = "Pitching Power", title = "Top 7 Teams by their Pitching Power in 2022") +
  theme_bw() +
  theme(legend.position = "none",
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x = element_blank()) +
  ylim(0, 1.025)


ggplot(batters, aes(xRV/n, xwOBA)) +
  geom_point() +
  stat_smooth(method = "lm") +
  geom_text(x = -0.004, y = 0.55, label = paste("R:", round(cor(batters$xRV/batters$n, batters$xwOBA), 3))) +
  theme_bw() +
  labs(x = "xRV per Pitch", title = "Quality of Pitching Faced vs Hitter Performance")

cor(batters$xRV/batters$n, batters$xwOBA)



ggplot(head(arrange(filter(batters, game_year == 2022), desc(RVd)), 5), aes(reorder(batter, -RVd), RVd, fill = batter)) +
  geom_col(color = "black") +
  geom_mlb_headshots(aes(player_id = c(592450, 502671, 518692, 607043, 592518)), height = 0.2) +
  scale_fill_manual(values = c("navy", "orange", "dodger blue", "brown", "red")) +
  theme_bw() +
  theme(legend.position = "none",
        axis.ticks.x = element_blank()) +
  ylim(0, 90) +
  labs(x = "Batters", y = "RCAE", title = "Top Batters in 2022 by RCAE")



ggplot(head(arrange(filter(batters, game_year == 2022), (RVd)), 5), aes(reorder(batter, -RVd), RVd, fill = batter)) +
  geom_col(color = "black") +
  geom_mlb_headshots(aes(player_id = c(570731, 664702, 670032, 54472, 595978)), height = 0.2) +
  scale_fill_manual(values = c("gold", "navy", "black", "red", "royal blue")) +
  theme_bw() +
  theme(legend.position = "none",
        axis.ticks.x = element_blank()) +
  ylim(-35, 0) +
  labs(x = "Batters", y = "RCAE", title = "Worst Batters in 2022 by RCAE")
