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

a
a <- subset(manual_data_one_control, (manual_data_one_control$ID %in% dfSorted$ID))
b <- subset(dfSorted, !(dfSorted$ID %in% manual_data_one_control$ID))            

### Drop all rows with comments in both dfs ###

### a = manual, dfSorted = automatic values###
##############################################

c <- subset(a, (is.na(a$comment) == TRUE))
d <- subset(dfSorted, (dfSorted$ID %in% c$ID)) 

boxplot(c$BL,d$Bodylength.µm.)
plot(dfSorted$Spinalength.µm.)

### This plots prob instead of frequency, x axis stilled called freq


c$Analysis <- "Manual"
d$Analysis <- "Automatic"
colnames(d)[colnames(d) == "Bodylength.µm."] ="BL"
colnames(d)[colnames(d) == "Spinalength.µm."] ="SL"
c <- c[order(c$ID),]
d <- d[order(c$ID),]
rownames(c) <- 1:nrow(c)
rownames(d) <- 1:nrow(d)
d$conc <- c[(c$ID %in% d$ID),]$conc
d$bile <- c[(c$ID %in% d$ID),]$bile
e <- rbind(c[c("treatment", "repl_NO", "animal", "BL", "SL", "ID", "Analysis","conc","bile")],d)
extremes <- e[e$SL> 1000,]
no_extremes <- e[!(e$ID %in% extremes$ID),]
library(ggplot2)

#####

e %>%
  ggplot(aes(Analysis, BL)) +
  geom_violin(aes(fill=Analysis)) +
  geom_point(width = 0.2, height = 1.5)+
  geom_line(aes(group = ID), col = "light grey", lty = "dashed") + theme_minimal()+ theme(text = element_text(size = 20)) +
  annotate(geom="text", x=1.5, y=2200, label= paste("n =",length(e$BL)/2, "/ 235"))+
  ylab("Body length [µm]") + ggtitle("Body length comparison | Marvin")
no_extremes %>%
  ggplot(aes(Analysis, SL)) +
  geom_violin(aes(fill=Analysis)) +
  geom_point(width = 0.2, height = 1.5)+
  geom_line(aes(group = ID), col = "light grey", lty = "dashed") + theme_minimal() + theme(text = element_text(size = 20)) +
  annotate(geom="text", x=1.5, y=270, label= paste("n =",length(no_extremes$SL)/2, "/ 235"))+
   ylab("Spina length [µm]") + ggtitle("Spina length comparison | Marvin")

ggsave("scatterplot_connecting_paired_points_with_lines_ggplot2.png")


###### Plot differences between the data points
###############################################
f <- subset(no_extremes,no_extremes$Analysis == "Manual")
f2 <- subset(no_extremes,no_extremes$Analysis == "Automatic")
Diff = merge(f,f2, by = "ID")
Diff$BL_Diff <- Diff$BL.y - Diff$BL.x 
Diff$SL_Diff <- Diff$SL.y - Diff$SL.x 
min(abs(Diff$BL_Diff), na.rm = TRUE)
max(abs(Diff$BL_Diff), na.rm = TRUE)
mean(abs(Diff$BL_Diff), na.rm = TRUE)

sd(subset(e, e$Analysis == "Automatic")$BL, na.rm = TRUE)
sd(subset(no_extremes, no_extremes$Analysis == "Automatic")$SL, na.rm = TRUE)
sd(subset(e, e$Analysis == "Manual")$BL, na.rm = TRUE)
sd(subset(e, e$Analysis == "Manual")$SL, na.rm = TRUE)

ggplot(e, aes(x = Analysis, y = SL)) +
  geom_point(aes(fill=ID))
#####


no_extremes <- no_extremes[order(no_extremes$ID),]
e <- e[order(e$ID),]
Df_SL_all <- Df_SL_all[order(Df_SL_all$Name),]
par(cex = 1.5)
plot(subset(e, e$Analysis == "Automatic")$BL,
     subset(e, e$Analysis == "Manual")$BL, 
     ylab = "Manual body length [µm]",
     xlab = "Automatic body length [µm]",
     main = "Automatic v. Manual body length",
     ylim = c(2200, 3000),
     xlim = c(2200,3000),
     frame.plot=FALSE)
      polygon(c(2000,3000,3000), c(2000,3000, 2000), col = "lightgrey", border = FALSE)
      abline(0,1, col = "red", lty = "dashed" )
      axis(2, at = c(0,3000))
      axis(1, at= c(0,3000))
      text(x = 2400, y = 2800, labels = "Manual > Automatic")
      text(x = 2800, y = 2400, labels = "Manual < Automatic")
      points(subset(e, e$Analysis == "Automatic")$BL,
             subset(e, e$Analysis == "Manual")$BL)    
      points(Df_BL$Paired_data.Bodylength.mm., 
             Df_BL$Paired_data.Körperlänge..µm., pch = 4, col = "blue")
      text(2250,2980, labels = paste("n =",length(subset(e, e$Analysis == "Automatic")$BL)))
      text(2350,2980, labels = paste("n =",length(Df_BL$Paired_data.Bodylength.mm.)), col = "blue")
        
      
      plot(subset(e, e$Analysis == "Automatic")$SL,
             subset(e, e$Analysis == "Manual")$SL, 
             ylab = "manual spina length [µm]",
             xlab = "automatic spina length [µm]",
             main = "Automatic v. manual spina length",
             ylim = c(200, 1000),
             xlim = c(200,3000),
             frame.plot=FALSE)
        polygon(c(100,3100,3100), c(100,100, 3100), col = "lightgrey", border = FALSE)
        abline(0,1, col = "red", lty = "dashed" )
        axis(2, at = c(-1,3000))
        axis(1, at= c(-1,3000))
        text(x = 400, y = 800, labels = "Manual > Automatic")
        text(x = 800, y = 400, labels = "Manual < Automatic")
        text(250,1000, labels = paste("n =",length(subset(e, e$Analysis == "Automatic")$SL)))
        ExtremesWithCounterpart <- e[e$ID %in% extremes$ID,]
        points(subset(e, e$Analysis == "Automatic")$SL,
               subset(e, e$Analysis == "Manual")$SL)    
        points(Df_SL$Paired_data.Spinalength.mm.,
               Df_SL$Paired_data.Spinalänge..µm., pch = 4, col = "blue")    
        text(500,1000, labels = paste("n =",length(Df_SL$Paired_data.Spinalength.mm.)), col = "blue")
        
        
        plot(subset(no_extremes, no_extremes$Analysis == "Automatic")$SL,
             subset(no_extremes, no_extremes$Analysis == "Manual")$SL, 
             ylab = "manual spina length [µm]",
             xlab = "automatic spina length [µm]",
             main = "Automatic v. manual spina length",
             ylim = c(200, 1000),
             xlim = c(200,1000),
             frame.plot=FALSE)
        polygon(c(100,1100,1100), c(100,100, 1100), col = "lightgrey", border = FALSE)
        abline(0,1, col = "red", lty = "dashed" )
        axis(2, at = c(-1,3000))
        axis(1, at= c(-1,3000))
        text(x = 400, y = 800, labels = "Manual > Automatic")
        text(x = 800, y = 400, labels = "Manual < Automatic")
        text(250,1000, labels = paste("n =",length(subset(no_extremes, no_extremes$Analysis == "Automatic")$SL)))
        points(subset(no_extremes, no_extremes$Analysis == "Automatic")$SL,
               subset(no_extremes, no_extremes$Analysis == "Manual")$SL)
        points(subset(Df_SL_all, Df_SL_all$Analysis == "Automatic")$SL,
               subset(Df_SL_all, Df_SL_all$Analysis == "Manual")$SL,
               pch = 4, col = "blue")    
        text(350,1000, labels = paste("n =",length(Df_SL$Paired_data.Spinalength.mm.)), col = "blue")
        
#rect(xleft = 2000, ybottom = 2000, xright = 3000, ytop = 3000, col="grey")
##Simonas data #######

SimonaManual <- read.csv2("~/GitRStudioServer/ImageData/SimonaRelabelled/annotations/SimonaAig21Measure.csv", encoding = "latin1")
SimonaMachine <- read.csv("~/GitRStudioServer/ImageData/SimonaRelabelled/annotations/Datafinished.csv")
typeof(SimonaMachine$Bodylength.mm.)
## Drop calibration
SimonaManual <- SimonaManual[-c(13,14), ]

SimonaMachine <- subset(SimonaMachine,SimonaMachine$distance_per_pixel < 0.0028)
SimonaManual <- subset(SimonaManual,SimonaMachine$distance_per_pixel < 0.0028)
SimonaManual$Bildname
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

### Bias plots



Df_SL$Manuel = "Manual"
Df_SL$Automatic = "Automatic"
Df_BL$Automatic = "Automatic"
Df_BL$Manuel = "Manual"

#### Make two Dfs to rbind them later
Df_SL_Manuel <- data.frame(Df_SL$Paired_data.Bildname, Df_SL$Paired_data.Spinalänge..µm., Df_SL$Manuel)
Df_SL_Automatic  <- data.frame(Df_SL$Paired_data.Bildname, Df_SL$Paired_data.Spinalength.mm., Df_SL$Automatic)
names <- c("Name", "SL", "Treatment")
colnames(Df_SL_Manuel) <- c("Name", "SL", "Analysis")
colnames(Df_SL_Automatic) <- c("Name", "SL", "Analysis")
Df_SL_all <- rbind(Df_SL_Manuel,Df_SL_Automatic)
SimonaMachine[SimonaMachine$Bildname == "Aig Cellulose 5000 83",]
SimonaManual[SimonaManual$Bildname == "Aig Cellulose 5000 83",]
Df_SL_all %>%
  ggplot(aes(Analysis, SL)) +
  geom_violin(aes(fill=Analysis)) +
  geom_point(width = 0.2, height = 1.5)+
  geom_line(aes(group = Name), col = "light grey", lty = "dashed") + theme_minimal() + theme(text = element_text(size = 20)) +
  annotate(geom="text", x=1.5, y=270, label= paste("n =",length(Df_SL_all$SL)/2, "/", length(SimonaManual$Spinalänge..µm.)))+
  xlab("Analysis") + ylab("Spina length [µm]") + ggtitle("Spina length comparison | Simona")



#### BL df
Df_BL_Manuel <- data.frame(Df_BL$Paired_data.Bildname, Df_BL$Paired_data.Körperlänge..µm., Df_BL$Manuel)
Df_BL_Automatic  <- data.frame(Df_BL$Paired_data.Bildname, Df_BL$Paired_data.Bodylength.mm., Df_BL$Automatic)
names <- c("Name", "BL", "Treatment")
colnames(Df_BL_Manuel) <- c("Name", "BL", "Analysis")
colnames(Df_BL_Automatic) <- c("Name", "BL", "Analysis")
Df_BL_all <- rbind(Df_BL_Manuel,Df_BL_Automatic)

Df_BL_all %>%
  ggplot(aes(Analysis, BL)) +
  geom_violin(aes(fill=Analysis)) +
  geom_point(width = 0.2, height = 1.5)+
  geom_line(aes(group = Name), col = "light grey", lty = "dashed") + theme_minimal() + theme(text = element_text(size = 20)) +
  annotate(geom="text", x=1.5, y=2200, label= paste("n =",length(Df_BL_all$BL)/2, "/", length(SimonaManual$Spinalänge..µm.)))+
  xlab("Analysis") + ylab("Body length [µm]") + ggtitle("Body length comparison | Simona")



##### Consttruction #####
no_extremes$logconc <- log10(no_extremes$conc)

no_extremes %>%
  ggplot(aes(treatment, BL)) +
  geom_boxplot(aes(fill=treatment)) +
  theme_minimal() +
  geom_vline(xintercept = c(8.5, 9.5))
  #geom_point(width = 0.2, height = 1.5)+
  #geom_line(aes(group = Name), col = "light grey", lty = "dashed") + theme_minimal() +
  #annotate(geom="text", x=1.5, y=2200, label= paste("n =",length(Df_BL_all$BL)/2, "/", length(SimonaManual$Spinalänge..µm.)))+
  #xlab("Analysis") + ylab("Body length [µm]") + ggtitle("Body length comparison | Simona")

no_extremes %>%
  ggplot(aes(x = logconc,y = SL)) +
  geom_point(aes(col=bile)) +
  theme_minimal() +
  xlab("log(conc)") +
  ylab("body lenght [µm]")+
  ggtitle(label = "Automatic body length")
  
  #geom_vline(xintercept = c(8.5, 9.5))

subset(e, e$Analysis == "Manual") %>%
  ggplot(aes(x = logconc,y = BL)) +
  geom_point(aes(col=bile)) +
  geom_smooth(aes(col = bile,fill=bile))+
  theme_classic()+
  xlab("log(conc)") +
  ylab("body lenght [µm]")+
  ggtitle(label = "Manual body length")

subset(e, e$Analysis == "Automatic") %>%
  ggplot(aes(x = logconc,y = BL)) +
  geom_point(aes(col=bile)) +
  geom_smooth(aes(col = bile,fill=bile))+
  theme_classic()+
  xlab("log(conc)") +
  ylab("body lenght [µm]")+
  ggtitle(label = "Automatic body length")

### make into Marvins format
OnlyDry <- subset(no_extremes, no_extremes$bile == "dry")

OnlyDry %>%
  ggplot(aes(x = logconc,y = SL)) +
  geom_point(aes(col= Analysis)) +
  geom_smooth(aes(col = Analysis,fill= Analysis))+
  theme_classic() +
  xlab("Log(conc)") +
  ylab("Spina length [µm]")+
  ggtitle(label =  "Spina length | dry bile")

OnlyDry %>%
  ggplot(aes(x = logconc,y = BL)) +
  geom_point(aes(col= Analysis)) +
  geom_smooth(aes(col = Analysis,fill= Analysis))+
  theme_classic() +
  theme(text = element_text(size = 20))+
  xlab("Log(conc)") +
  ylab("Body length [µm]")+
  ggtitle(label = "Body length | dry bile")

#### Make t.test###
as.factor()
Controls <- subset(no_extremes, no_extremes$treatment == "C")
MaxConc <- subset(OnlyDry, OnlyDry$treatment == "1µM")

t.test(subset(Controls, Controls$Analysis == "Manual")$BL, subset(MaxConc, MaxConc$Analysis == "Manual")$BL)
t.test(subset(Controls, Controls$Analysis == "Automatic")$BL, subset(MaxConc, MaxConc$Analysis == "Automatic")$BL)

t.test(subset(Controls, Controls$Analysis == "Manual")$SL, subset(MaxConc, MaxConc$Analysis == "Manual")$SL)
t.test(subset(Controls, Controls$Analysis == "Automatic")$SL, subset(MaxConc, MaxConc$Analysis == "Automatic")$SL)

BoxData <- rbind(Controls,MaxConc)
BoxData$Identify <- paste(BoxData$Analysis,BoxData$conc)
BoxData %>%
  ggplot(aes(x = Identify,y = BL, fill = as.factor(Identify))) +
  geom_violin()+
  geom_boxplot(width = 0.1)+
  theme_classic() +
  theme(text = element_text(size = 20),legend.position = "none") +
  scale_fill_manual(values = c("#F8766D", "#f8bc6d", "#00BFC4", "#619CFF"))+
  xlab("Treatment") +
  ylab("Body length [µm]")+
  ggtitle(label = "Body length | dry bile")+
  geom_segment(aes(y = 2850, x = 1, yend = 2850, xend = 2))+
  geom_segment(aes(y = 2850, x = 3, yend = 2850, xend = 4)) +
  scale_x_discrete(labels=c("Control automatic","1 µM automatic","Control manual","1 µM manual"))+
  annotate(geom = "text",x = 1.5, y = 2865, label ="p = 0.45")+
  annotate(geom = "text",x = 3.5, y = 2865, label ="p = 0.51")+
  geom_vline(xintercept = 2.5)
  
mean(subset(BoxData, BoxData$Identify == "Automatic 1")$BL) 
##### ANCOVA between the lines ####
###################################

f3 <- subset(Df_SL_all,Df_SL_all$Analysis == "Manual")
f4 <- subset(Df_SL_all,Df_SL_all$Analysis == "Automatic")
Diff = merge(f3 ,f4 , by = "Name")
Diff$SL_Diff <- Diff$SL.y - Diff$SL.x
max(abs(Diff$SL_Diff), na.rm = TRUE)
min(abs(Diff$SL_Diff), na.rm = TRUE)
mean(abs(Diff$SL_Diff))
sd(subset(Df_SL_all,Df_SL_all$Analysis == "Manual")$SL, na.rm = TRUE)
sd(subset(Df_SL_all,Df_SL_all$Analysis == "Automatic")$SL, na.rm = TRUE)
sd(subset(Df_BL_all,Df_SL_all$Analysis == "Manual")$BL, na.rm = TRUE)
sd(subset(Df_BL_all,Df_SL_all$Analysis == "Automatic")$BL, na.rm = TRUE)
f5 <- subset(Df_BL_all,Df_BL_all$Analysis == "Manual")
f6 <- subset(Df_BL_all,Df_BL_all$Analysis == "Automatic")
Diff = merge(f5 ,f6 , by = "Name")
Diff$BL_Diff <- Diff$BL.y - Diff$BL.x
max(abs(Diff$BL_Diff), na.rm = TRUE)
min(abs(Diff$BL_Diff), na.rm = TRUE)
mean(abs(Diff$BL_Diff))


### levene not necessary
install.packages("car")
library(car)

leveneTest(BL ~ Analysis,e)
leveneTest(SL ~ Analysis,no_extremes)
leveneTest(BL ~ Analysis,Df_BL_all)
leveneTest(SL ~ Analysis,Df_SL_all)


### trash

geom_segment(aes(y = median(subset(BoxData, BoxData$Identify == "Automatic 0")$BL), x = 1, 
                 yend = median(subset(BoxData, BoxData$Identify == "Automatic 0")$BL), xend = 1.5), lty = 'dashed')+
  geom_segment(aes(y = median(subset(BoxData, BoxData$Identify == "Automatic 1")$BL), x = 2, 
                   yend = median(subset(BoxData, BoxData$Identify == "Automatic 1")$BL), xend = 1.5), lty = 'dashed')+
  geom_segment(aes(y = median(subset(BoxData, BoxData$Identify == "Manual 0")$BL), x = 3, 
                   yend = median(subset(BoxData, BoxData$Identify == "Manual 0")$BL), xend = 3.5), lty = 'dashed')+
  geom_segment(aes(y = median(subset(BoxData, BoxData$Identify == "Manual 1")$BL), x = 4, 
                   yend = median(subset(BoxData, BoxData$Identify == "Manual 1")$BL), xend = 3.5), lty = 'dashed')+