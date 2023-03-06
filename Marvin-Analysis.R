##### Data of Marvin Data

automatic_data <- read.csv2("Marvin-Validation/annotations4.csv")

library("readxl")
manual_data <- read_excel("Marvin-Validation/Test fish bile magna Aig 3.xlsx", sheet = 3)
manual_data <- data.frame(manual_data)

#### To get the same indexes and names we have to split the rows of automatic data
#### Drop index Row

automatic_data <- automatic_data[-c(1)]
library(stringr)


automatic_data[c('temporary', 'animal')] <- str_split_fixed(automatic_data$image_id, '_', 2)

# Rearrange columns and remove original name column
df <- automatic_data[c('temporary', 'animal', 'Bodylength.µm.', "Spinalength.µm.")]
df[c("animal", "Image_format")] <- str_split_fixed(df$animal, "[.]", 2)
df <- df[c('temporary', 'animal', 'Bodylength.µm.', "Spinalength.µm.")]
dfSorted <- df[order(df$temporary),]

dfSorted$treatment <- str_sub(dfSorted$temporary, end=-2)
dfSorted$repl_NO <- str_sub(dfSorted$temporary, - 1, - 1) 
# Rearrange columns and remove original name column
dfSorted <- dfSorted[c('treatment', 'repl_NO', 'animal', 'Bodylength.µm.', "Spinalength.µm.")]
dfSorted$repl_NO <- as.integer(dfSorted$repl_NO)
dfSorted$animal <- as.integer(dfSorted$animal)

## Now we have to adjust the names add fb to every entry and move the f from the beginning to the end


manual_data_one_control <- subset(manual_data,manual_data$treatment != "Cf")

length(manual_data_one_control$SL)
manual_data_one_control[order(manual_data_one_control$treatment),]

### To find our discrepancies in the dataset we make a unqiue Row which is name, repl_No, treatment
### Drop the calibration row

dfSorted <- subset(dfSorted, dfSorted$treatment != "kalib.jp")

### Make dfs only containing matching IDs
manual_data_one_control$ID <- paste(manual_data_one_control$treatment,manual_data_one_control$repl_NO,manual_data_one_control$animal)
dfSorted$ID <- paste(dfSorted$treatment, dfSorted$repl_NO,dfSorted$animal)


a <- subset(manual_data_one_control, (manual_data_one_control$ID %in% dfSorted$ID))
b <- subset(dfSorted, !(dfSorted$ID %in% manual_data_one_control$ID))            

### Drop all rows with comments in both dfs ###

### a = manual, dfSorted = automatic values###
##############################################

c <- subset(a, (is.na(a$comment) == TRUE))
d <- subset(dfSorted, (dfSorted$ID %in% c$ID)) 

boxplot(c$BL,d$Bodylength.µm.)
plot(dfSorted$Spinalength.µm.)

t.test(c$SL,d$Spinalength.µm., paired = TRUE)
t.test(c$BL,d$Bodylength.µm., paired = TRUE)
chisq.test(c$BL,d$Bodylength.µm.)
### This plots prob instead of frequency, x axis stilled called freq


c$Analysis <- "Manual_Marvin"
d$Analysis <- "Automatic_Marvin"
colnames(d)[colnames(d) == "Bodylength.µm."] ="BL"
colnames(d)[colnames(d) == "Spinalength.µm."] ="SL"
c <- c[order(c$ID),]
d <- d[order(c$ID),]
rownames(c) <- 1:nrow(c)
rownames(d) <- 1:nrow(d)
e <- rbind(c[c("treatment", "repl_NO", "animal", "BL", "SL", "ID", "Analysis")],d)

library(ggplot2)

ggplot(no_extremes, aes(x = BL, y = Analysis)) +
  geom_violin(trim=FALSE, fill='#A4A4A4', color="darkred")+
  geom_boxplot(width=0.1) + theme_minimal()

ggplot(e, aes(x = SL, y = Analysis)) +
  geom_violin(trim=FALSE, fill='#A4A4A4', color="darkred")+
  geom_boxplot(width=0.1) + theme_minimal()

#####
library(ggstatsplot)

ggbetweenstats(
  data  = e,
  x     = Analysis,
  y     = BL,
  grouping.var = ID,
  title = "Distribution of body length between evaluations"
)

no_extremes %>%
  ggplot(aes(Analysis, BL)) +
  geom_violin(aes(fill=Analysis)) +
  geom_point(width = 0.2, height = 1.5)+
  geom_line(aes(group = ID), col = "light grey", lty = "dashed") + theme_minimal()
ggsave("scatterplot_connecting_paired_points_with_lines_ggplot2.png")


ggbetweenstats(
  data  = e,
  x     = Analysis,
  y     = SL,
  grouping.var = ID,
  title = "Distribution of spina length between evaluations"
)

#### for visualization drop the 5 outliers

extremes <- e[e$SL> 1000,]

no_extremes <- e[!(e$ID %in% extremes$ID),]

ggbetweenstats(
  data  = no_extremes,
  x     = Analysis,
  y     = SL,
  grouping.var = ID,
  title = "Distribution of spina length between evaluations"
)

###### Plot differences between the data points
###############################################
f <- subset(no_extremes,no_extremes$Analysis == "Manual_Marvin")
f2 <- subset(no_extremes,no_extremes$Analysis == "Automatic_Marvin")
Diff = merge(f,f2, by = "ID")
Diff$BL_Diff <- Diff$BL.y - Diff$BL.x 
Diff$SL_Diff <- Diff$SL.y - Diff$SL.x 
max(Diff$BL_Diff, na.rm = TRUE)
min(Diff$BL_Diff, na.rm = TRUE)

max(Diff$SL_Diff, na.rm = TRUE)
min(Diff$SL_Diff, na.rm = TRUE)

ggplot(e, aes(x = Analysis, y = SL)) +
  geom_point(aes(fill=ID))
#####
no_extremes <- no_extremes[order(no_extremes$ID),]
e <- e[order(e$ID),]
plot(subset(e, e$Analysis == "Automatic_Marvin")$BL,
     subset(e, e$Analysis == "Manual_Marvin")$BL, 
     ylab = "body length manually [µm]",
     xlab = "body length automatic [µm]",
     main = "Automatic v. Manual body length Marvin",
     ylim = c(2200, 3000),
     xlim = c(2200,3000),
     frame.plot=FALSE)
      polygon(c(2000,3000,3000), c(2000,3000, 2000), col = "lightgrey", border = FALSE)
      abline(0,1, col = "red", lty = "dashed" )
      axis(2, at = c(0,3000))
      axis(1, at= c(0,3000))
      text(x = 2400, y = 2800, labels = "Manual < Automatic")
      text(x = 2800, y = 2400, labels = "Manual > Automatic")
      points(subset(e, e$Analysis == "Automatic_Marvin")$BL,
             subset(e, e$Analysis == "Manual_Marvin")$BL)    
      points(Df_BL$Paired_data.Bodylength.mm., 
             Df_BL$Paired_data.Körperlänge..µm., pch = 4, col = "blue")
      text(2250,2980, labels = paste("n =",length(subset(e, e$Analysis == "Automatic_Marvin")$BL)))
      text(2350,2980, labels = paste("n =",length(Df_BL$Paired_data.Bodylength.mm.)), col = "blue")
        
      
      plot(subset(e, e$Analysis == "Automatic_Marvin")$SL,
             subset(e, e$Analysis == "Manual_Marvin")$SL, 
             ylab = "spina length manually [µm]",
             xlab = "spina length automatic [µm]",
             main = "Automatic v. Manual spina length Marvin",
             ylim = c(200, 1000),
             xlim = c(200,3000),
             frame.plot=FALSE)
        polygon(c(100,3100,3100), c(100,100, 3100), col = "lightgrey", border = FALSE)
        abline(0,1, col = "red", lty = "dashed" )
        axis(2, at = c(-1,3000))
        axis(1, at= c(-1,3000))
        text(x = 400, y = 800, labels = "Manual < Automatic")
        text(x = 800, y = 400, labels = "Manual > Automatic")
        text(250,1000, labels = paste("n =",length(subset(e, e$Analysis == "Automatic_Marvin")$SL)))
        ExtremesWithCounterpart <- e[e$ID %in% extremes$ID,]
        points(subset(e, e$Analysis == "Automatic_Marvin")$SL,
               subset(e, e$Analysis == "Manual_Marvin")$SL)    
        points(Df_SL$Paired_data.Spinalength.mm.,
               Df_SL$Paired_data.Spinalänge..µm., pch = 4, col = "blue")    
        text(500,1000, labels = paste("n =",length(Df_SL$Paired_data.Spinalength.mm.)), col = "blue")
        
        
        plot(subset(no_extremes, no_extremes$Analysis == "Automatic_Marvin")$SL,
             subset(no_extremes, no_extremes$Analysis == "Manual_Marvin")$SL, 
             ylab = "spina length manually [µm]",
             xlab = "spina length automatic [µm]",
             main = "Automatic v. Manual spina length Marvin",
             ylim = c(200, 1000),
             xlim = c(200,1000),
             frame.plot=FALSE)
        polygon(c(100,1100,1100), c(100,100, 1100), col = "lightgrey", border = FALSE)
        abline(0,1, col = "red", lty = "dashed" )
        axis(2, at = c(-1,3000))
        axis(1, at= c(-1,3000))
        text(x = 400, y = 800, labels = "Manual < Automatic")
        text(x = 800, y = 400, labels = "Manual > Automatic")
        text(250,1000, labels = paste("n =",length(subset(e, e$Analysis == "Automatic_Marvin")$SL)))
        points(subset(no_extremes, no_extremes$Analysis == "Automatic_Marvin")$SL,
               subset(no_extremes, no_extremes$Analysis == "Manual_Marvin")$SL)
        points(Df_SL$Paired_data.Spinalength.mm.,
               Df_SL$Paired_data.Spinalänge..µm., pch = 4, col = "blue")    
        text(350,1000, labels = paste("n =",length(Df_SL$Paired_data.Spinalength.mm.)), col = "blue")
        
#rect(xleft = 2000, ybottom = 2000, xright = 3000, ytop = 3000, col="grey")
##Simonas data #######

SimonaManual <- read.csv2("~/GitRStudioServer/ImageData/SimonaRelabelled/annotations/SimonaAig21Measure.csv", encoding = "latin1")
SimonaMachine <- read.csv("~/GitRStudioServer/ImageData/SimonaRelabelled/annotations/Datafinished.csv")
typeof(SimonaMachine$Bodylength.mm.)
## Drop calibration
SimonaManual <- SimonaManual[-c(13,14), ]
SimonaMachine$Bodylength.mm. <- SimonaMachine$Bodylength.mm.*1000 ## make same unit
SimonaMachine$Spinalength.mm. <- SimonaMachine$Spinalength.mm.*1000

names(SimonaMachine)[2] = "Bildname"
SimonaMachine$Bildname <- sapply(strsplit(SimonaMachine$Bildname, split='.', fixed=TRUE), function(x) (x[1]))
SimonaManual$Bildname  <- sapply(strsplit(SimonaManual$Bildname, split='.', fixed=TRUE), function(x) (x[1]))

## Change name to merge using the titel

Paired_data <- merge(SimonaMachine,SimonaManual, by = "Bildname")
Df_essential 
Df_SL <-data.frame(Paired_data$Bildname,Paired_data$Spinalänge..µm.,Paired_data$Spinalength.mm.)
Df_BL <- data.frame(Paired_data$Bildname,Paired_data$Körperlänge..µm.,Paired_data$Bodylength.mm.)
Df_SL <- na.omit(Df_SL)
Df_BL <- na.omit(Df_BL)
class(Df_SL)
### Change into ggplot format ### Change this up merge by Bildname
Manuel <- rep("Manuel", length(Testdf$Paired_data.Bildname))
Automatisch <- rep("Automatisch", length(Testdf$Paired_data.Bildname))
paired <- rep(1:length(Testdf$Paired_data.Bildname))
Treatments <- c(Manuel, Automatisch)
Pairings <- c(paired,paired)
BL <- c(Testdf$Paired_data.Körperlänge..µm.,Testdf$Paired_data.Bodylength.mm.)
SL <- c(Testdf$Paired_data.Spinalänge..µm.,Testdf$Paired_data.Spinalength.mm.)
Diff <- Testdf$Paired_data.Körperlänge..µm. - Testdf$Paired_data.Bodylength.mm.
ggdf = data.frame(Treatments,Messungen,BL,SL,Diff)


### Alle ausreißer hohe conv factors
subset(Paired_data, Paired_data$Bildname == "Aig PET 5000 18")
subset(Paired_data, Paired_data$Bildname == "Aig PET 500 7")
subset(Paired_data, Paired_data$Bildname == "Aig PET 500 5")
Paired_data$Spinalength.mm.
ggbetweenstats(
  data  = ggdf,
  x     = Treatments,
  y     = BL,
  grouping.var = ID,
  title = "Distribution of body length between evaluations - Simona"
)

ggbetweenstats(
  data  = ggdf,
  x     = Treatments,
  y     = SL,
  grouping.var = ID,
  title = "Distribution of spina length between evaluations - Simona"
)



### Bias plots
chisq.test(Df_BL$Paired_data.Bodylength.mm.,Df_BL$Paired_data.Körperlänge..µm.)
chisq.test(Df_SL$Paired_data.Spinalength.mm., Df_SL$Paired_data.Spinalänge..µm.)

plot(Df_BL$Paired_data.Bodylength.mm., 
     Df_BL$Paired_data.Körperlänge..µm.,
     ylab = "body length manually [µm]",
     xlab = "body length automatic [µm]",
     main = "Automatic v. Manual body length Simona",
     ylim = c(2200, 3000),
     xlim = c(2200,3000),
     frame.plot=FALSE)
polygon(c(2000,3000,3000), c(2000,3000, 2000), col = "lightgrey", border = FALSE)
abline(0,1, col = "red", lty = "dashed" )
axis(2, at = c(0,3000))
axis(1, at= c(0,3000))
text(x = 2400, y = 2800, labels = "Manual < Automatic")
text(x = 2800, y = 2400, labels = "Manual > Automatic")
points(Df_BL$Paired_data.Bodylength.mm., 
       Df_BL$Paired_data.Körperlänge..µm.)   
text(2250,2980, labels = paste("n =",length(Df_BL$Paired_data.Bodylength.mm.)))


plot(Df_SL$Paired_data.Spinalength.mm.,
     Df_SL$Paired_data.Spinalänge..µm., 
     ylab = "spina length manually [µm]",
     xlab = "spina length automatic [µm]",
     main = "Automatic v. Manual spina length Simona",
     ylim = c(200, 1000),
     xlim = c(200,1000),
     frame.plot=FALSE)
polygon(c(100,1100,1100), c(100,100, 1100), col = "lightgrey", border = FALSE)
abline(0,1, col = "red", lty = "dashed" )
axis(2, at = c(-1,3000))
axis(1, at= c(-1,3000))
text(x = 400, y = 800, labels = "Manual < Automatic")
text(x = 800, y = 400, labels = "Manual > Automatic")
points(Df_SL$Paired_data.Spinalength.mm.,
       Df_SL$Paired_data.Spinalänge..µm.)    
text(250,1000, labels = paste("n =",length(Df_SL$Paired_data.Spinalength.mm.)))

SimonaMachine