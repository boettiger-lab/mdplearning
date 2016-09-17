library("dplyr")
zero_is_na <- function(x){
  if(x[length(x)] > 0){
    x[x == 0] <- as.numeric(NA)
  }
  x
}



values     <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/values.csv")
assessment <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/assessment.csv")
stock      <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/stock.csv")
units      <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/units.csv")
area       <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/area.csv")
lmestock   <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/lmestock.csv")
lmerefs    <- read.csv("http://berkeley.carlboettiger.info/espm-88b/fish/data/lmerefs.csv")

tbl <-
  values %>%
  left_join(assessment) %>%
  left_join(stock) %>%
  left_join(units) %>%
  left_join(area) %>%
  left_join(lmestock) %>%
  left_join(lmerefs) %>%
  select(scientificname, commonname, tsyear, r, ssb, total, catch_landings,
         r_unit, ssb_unit, total_unit, catch_landings_unit, country, lme_name,
         lme_number, stockid, assessid) %>%

  group_by(commonname, lme_name, tsyear) %>%
  filter(catch_landings_unit == 'MT') %>%
  filter(total_unit == 'MT') %>%
  filter(lme_name == "Pacific High Seas") %>%
  filter(commonname == "Southern bluefin tuna") %>%
  summarise_at(vars(total, catch_landings), sum, na.rm=FALSE) -> fish

fish$catch_landings <- zero_is_na(fish$catch_landings)
fish <- na.omit(fish)
bluefin_tuna <- fish
devtools::use_data(bluefin_tuna)


N <- dim(fish)[1]
scale <- max(fish$total)
scaled_data <- data.frame(t = 1:N, y = fish$total / scale, a = fish$catch_landings / scale)

devtools::use_data(scaled_data)
