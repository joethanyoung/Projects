# version_control 1.1
# 修改了单飞提出的审计程序（1）投保人有重名的情况
# version 1.2



library(tictoc())
#
tic()

# load libraries
library(readxl)
library(dplyr)
library(writexl)
library(lubridate)
library(stringr)
library(pROC)
library(openxlsx)
library(tidyr)

# load the dataframe
df <-
  read_excel(
    "L01105基础表个险承保清单.xlsx", 
    skip = 1
    )

# convert cbrq to as date
df$承保日期 <- as.Date(
  df$承保日期, 
  format = "%Y-%m-%d"
  )

# Check for missing values in relevant columns
if (any(is.na(
  df$投保人姓名), is.na(df$险种名称), is.na(df$承保日期))
  ) {
  print("Data contains missing values")
}

# na count the colsums
na_count <- colSums(
  is.na(
    df[c("投保人姓名", "险种名称", "承保日期", "投保人客户号")])
  )


# get names from dataframe
columns_without_na <- names(na_count)[na_count == 0]


#  filter out nas in 投保人、险种、承保日期
df <- df[complete.cases(
  df$投保人姓名, df$险种名称, df$承保日期, df$投保人客户号), ]

# filter out all the
df <- df[complete.cases(
  df[c("投保人姓名", "险种名称", "承保日期", "投保人客户号")]), ]

df$保单保额 <- as.numeric(df$保单保额)

# Create a new dataframe with counts of short period policies
df_count <- df %>%
  arrange(投保人姓名, 投保人客户号, 险种名称, 承保日期) %>%
  group_by(投保人姓名, 投保人客户号, 险种名称) %>%
  mutate(
    prev_date = lag(承保日期),
    date_diff = as.numeric(difftime(承保日期, prev_date, units = "days")),
    short_period_policies = ifelse(date_diff <= 30, 1, 0)
  ) %>%
  summarise(
    short_period_count = sum(short_period_policies, na.rm = TRUE),
    .groups = "drop"
  )

# Filter out policyholders with less than 2 short period policies and order by count
df_count <-
  df_count %>%
  filter(short_period_count >= 2) %>%
  arrange(-short_period_count)

# left join
df_join <- left_join(
  df_count, df, by = c("投保人姓名" = "投保人姓名", "投保人客户号" = "投保人客户号")
  )
df_join_distinct <- df_join %>% distinct(投保人姓名, 投保人客户号, .keep_all = TRUE)
df_join_distinct %>% rename(投保次数 = short_period_count)


# Find rows in df_join that are not in df_join_distinct
diff_df <- anti_join(
  df_join, df_join_distinct, by = c("投保人姓名", "投保人客户号")
  )

# Save the result to an Excel file
df_join_distinct %>% write_xlsx(
  "序号10_营运_承保_分析程序(1).xlsx"
  )

# Print the difference
print(diff_df)
#
# ##################################################################### index 13

# import dataframe
df2 <-
  read_excel("L06110基础表投保单清单.xlsx", skip = 1)

# filter out both 1 and 2
df2_ab <- df2 %>% filter(投保单状态 == "撤件" & 是否为自保件 == "否")

# convert slrq using as.date in standard format
df2_ab$受理日期 <- as.Date(df2_ab$受理日期, format = "%Y-%m-%d")

# classify dataframe into different layer
audit_periods <- unique(na.omit(df2_ab$受理日期))

# Convert the column to numeric before the loop
df2_ab$主副险合计保费 <- as.numeric(as.character(df2_ab$主副险合计保费))

# convert time
df2_ab$month <- format(df2_ab$受理日期, "%Y-%m")

# Define the unique months
months <- unique(df2_ab$month)

# Define the sampling percentage
sampling_percentage <- 0.01

# Initialize an empty dataframe
sampled_df <- data.frame()

# Loop through months
for (month in months) {
  month_df <- df2_ab %>%
    filter(month == !!month) %>%
    mutate(prob = 主副险合计保费 / sum(主副险合计保费)) %>%
    replace_na(list(prob = 0)) # replace NA probabilities with 0

  if (nrow(month_df) == 0) {
    print(paste("No records for month:", month))
    next
  }

  n <- max(round(nrow(month_df) * sampling_percentage), 1)
  n <- min(n, nrow(month_df))

  sampled_month_df <-
    month_df[sample(nrow(month_df), size = n, prob = month_df$prob), ]
  sampled_df <- rbind(sampled_df, sampled_month_df)
}

# Export to .xlsx file
sampled_df %>% write_xlsx("序号10_营运_承保_分析程序(2).xlsx")

# ##################################################################### index 14

# import dataframe
df3 <- read_excel("L06108基础表电话服务疑似异常业务提示工单清单.xlsx", skip = 1)

# take a glimpse
df3 %>% glimpse()

# filter out the dataframe
df3 <- df3 %>% filter(转办原因 == "重复号码回访成功疑似异常业务工单")

# output xlsx dataframe
df3 %>% write_xlsx("序号10_营运_承保_分析程序(3).xlsx")

# ##################################################################### index 42

# import the data
df4 <-
  read_excel(
    "L06105基础表理赔清单.xlsx", skip = 1
    )

# check if there are any nas
if (any(is.na(df4$总给付金额), is.na(df4$事故日期), is.na(df4$承保日期))) {
  print("Data contains missing values")
}

# find nas in listed columns
na_count <- colSums(is.na(df4[c("总给付金额", "事故日期", "承保日期")]))
na_count %>% print() # print the list

# filter out nas in listed columns
df4 <- df4[complete.cases(df4$总给付金额, df4$事故日期, df4$承保日期), ]
df4 <- df4 %>% filter(理赔结论 %in% c("赔付", "通融", "协议"))

# filter the data detect string has 主险
filtered_data <- df4 %>%
  filter(str_detect(主险名称, "康悦人生")) %>%
  mutate(
    承保日期 = parse_date_time(承保日期, orders = c("ymd", "ymd HMS")),
    事故日期 = parse_date_time(事故日期, orders = c("ymd", "ymd HMS")),
    记账日期 = parse_date_time(记账日期, orders = c("ymd", "ymd HMS")),
    总给付金额 = as.numeric(总给付金额)
  )

# filter the data and group by 保单号 using summarise and ignore all the values are null
payment_counts <- filtered_data %>%
  group_by(保单号, 给付责任) %>%
  summarise(赔付次数 = n(), 累计总给付金额 = sum(总给付金额, na.rm = TRUE)) %>%
  mutate(是否涉及提前给付 = ifelse(
    str_detect(给付责任, "提前给付保险金") | str_detect(给付责任, "重大疾病提前给付"),
    "是",
    "否"
  ))

# arrange order by 赔付次数
top_payment_cases <- payment_counts %>%
  arrange(desc(赔付次数))

# Calculate the 90th percentile
percentile_90 <- quantile(top_payment_cases$赔付次数, 0.9)

# print top payment cases
top_payment_cases <-
  top_payment_cases %>% 
  filter(赔付次数 >= percentile_90)

# output dataframe to xlsx
top_payment_cases %>% write_xlsx("序号23_营运_理赔_分析程序(1)(5).xlsx")

# ############################################################### index 43 & 47

# import dataframe & skip the first row
df5 <- read_excel("L06105基础表理赔清单.xlsx", skip = 1)

# mutate several columns
df5 <- df5 %>%
  mutate(
    承保日期 = parse_date_time(承保日期, orders = c("ymd", "ymd HMS")),
    受理日期 = parse_date_time(受理日期, orders = c("ymd", "ymd HMS")),
    事故日期 = parse_date_time(事故日期, orders = c("ymd", "ymd HMS"))
  )

# filter out the dataframe where 受理日期 and/or 事故日期 are NOT within the policy year of 承保日期
df5_different_year <- df5 %>%
  filter(
    year(受理日期) < year(承保日期) | year(受理日期) > year(承保日期) + 1 |
      year(事故日期) < year(承保日期) | year(事故日期) > year(承保日期) + 1
  )

# detect relevant strings and filter out
df5_different_year <- df5_different_year %>%
  mutate(是否无理赔优惠 = ifelse(str_detect(理赔结论描述, "无理赔优惠"), "是", "否"))

# output the dataframe
df5_different_year %>% write_xlsx("序号23_营运_理赔_分析程序(2)(6).xlsx")


# ##################################################################### index 49

# looping through
current_wd <- getwd()

# Looping through
all_files <- list.files(path = current_wd, pattern = ".xlsx$")

# loop through all the files with specified name-tag
shishou_files <- all_files[grepl("^L05103基础表实收清单", all_files)]
yingshou_files <- all_files[grepl("^L05106基础表续期总应收清单", all_files)]
lipei_files <- all_files[grepl("^L06105基础表理赔清单", all_files)]

# convert list to dataframe
shishou_df <-
  lapply(shishou_files, function(x) {
    read_excel(x, skip = 1)
  })
shishou_df <- do.call(rbind, shishou_df)

# convert list to dataframe
yingshou_df <-
  lapply(yingshou_files, function(x) {
    read_excel(x, skip = 1)
  })
yingshou_df <- do.call(rbind, yingshou_df)

# convert list to ddataframe
lipei_df <- lapply(lipei_files, function(x) {
  read_excel(x, skip = 1)
})
lipei_df <- do.call(rbind, lipei_df)

# convert column to data format
lipei_df$事故日期 <- as.Date(lipei_df$事故日期, format = "%Y-%m-%d")
yingshou_df$应缴日 <- as.Date(yingshou_df$应缴日, format = "%Y-%m-%d")

# join dataframe
yingshou_df_max <- yingshou_df %>%
  group_by(保单号) %>%
  summarise(应缴日 = max(应缴日, na.rm = TRUE))

# based on lipei_df using lef_join
result_df <- lipei_df %>%
  left_join(yingshou_df_max, by = "保单号")

# filter out the na values
result_df <- result_df[complete.cases(result_df$应缴日), ]

# create new column
result_df <- result_df %>%
  mutate(宽末日 = 应缴日 + days(60))

# filter out
filtered_df_ly <- result_df %>%
  filter(事故日期 >= 应缴日 & 事故日期 <= 宽末日)

# convert to date
filtered_df_ly$受理日期 <-
  as.Date(filtered_df_ly$受理日期, format = "%Y-%m-%d")

# setting up 宽限期
filtered_df_ly <- filtered_df_ly %>%
  mutate(宽限期 = ifelse(受理日期 >= 应缴日 &
    受理日期 <= 宽末日, "宽限期内受理", "宽限期外受理"))

# convert certain column to date
shishou_df$应缴日 <- as.Date(shishou_df$应缴日, format = "%Y-%m-%d")

# keep all the unique values
shishou_df <- shishou_df %>%
  distinct(保单号, 应缴日, 险种名称, .keep_all = TRUE)

# left join
matched_df <-
  left_join(filtered_df_ly, shishou_df, by = c("保单号", "应缴日", "主险名称" = "险种名称"))

# manipulation
matched_df <- matched_df %>%
  mutate(是否已理赔未缴费 = ifelse(is.na(付费金额.y), "是", "否")) %>%
  filter(是否已理赔未缴费 == "是")

# output the results
matched_df %>% write_xlsx("序号23_营运_理赔_分析程序(4)(8).xlsx")

#
toc()
