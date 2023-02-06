#### Simona eval for poster

SimonaManual <- read.csv2("SimonaAig21Measure.csv", encoding = "latin1")
SimonaMachine <- read.csv("instances_default.csv")

## Drop calibration
SimonaManual
SimonaManual <- SimonaManual[-c(13,14), ]
SimonaManual$Körperlänge..µm. <- SimonaManual$Körperlänge..µm./1000 ## make same unit
SimonaManual$Spinalänge..µm. <- SimonaManual$Spinalänge..µm./1000

chisq.test(SimonaManual$Körperlänge..µm., SimonaMachine$Spinalength.mm.)
boxplot(SimonaManual$Körperlänge..µm.,SimonaMachine$Bodylength.mm.)
### change .tif and .jpg to make the names able to match

SimonaMachine$Bildname <- sapply(strsplit(SimonaMachine$Bildname, split='.', fixed=TRUE), function(x) (x[1]))
SimonaManual$Bildname  <- sapply(strsplit(SimonaManual$Bildname, split='.', fixed=TRUE), function(x) (x[1]))

## Change name to merge using the titel
names(SimonaMachine)[2] = "Bildname"
install.packages("ggplot2")
install.packages("ggpubr")
library(ggplot2)
library(ggpubr)
Paired_data <- merge(SimonaMachine,SimonaManual, by = "Bildname")
Testdf = data.frame(Paired_data$Bildname,Paired_data$Körperlänge..µm.,Paired_data$Bodylength.mm.)

Testdf



### Change into ggplot format
Manuel <- rep("Manuel", length(Testdf$Paired_data.Bildname))
Automatisch <- rep("Automatisch", length(Testdf$Paired_data.Bildname))
paired <- rep(1:length(Testdf$Paired_data.Bildname))
Treatments <- c(Manuel, Automatisch)
Pairings <- c(paired,paired)
Messungen <- c(Testdf$Paired_data.Körperlänge..µm.,Testdf$Paired_data.Bodylength.mm.)
Diff <- Testdf$Paired_data.Körperlänge..µm. - Testdf$Paired_data.Bodylength.mm.

#### Set all negatives -1 and all pos. 1
Diff[Diff<0] = 0
Diff[Diff>0] = 1

## Diff is pos if Machine is longer than Manuel
ggdf = data.frame(Treatments,Messungen,Pairings,Diff)
ggdfNegative <- ggdf[ggdf$Diff < 0,]
ggdfPositive <- ggdf[ggdf$Diff > 0,] 
ggplot(data = ggdf,aes( x = Treatments,y = Messungen, fill= Treatments)) +
geom_violin() +
geom_point()+ 
geom_line(data = ggdfNegative, aes(group=Pairings), colour = "light blue") +
geom_line(data = ggdfPositive, aes(group=Pairings), colour = "orange") +
ylim(0,5) + ylab("body length [mm]") + 
theme_classic()

class(ggdf)

mean(Paired_data$Spinalänge..µm.)
mean(Paired_data$Spinalength.mm., na.rm = TRUE)

install.packages("gplots")
library("gplots")
mean(Testdf$Paired_data.Bodylength.mm., na.rm = TRUE)
mean(Testdf$Paired_data.Körperlänge..µm.)
boxplot2(Testdf$Paired_data.Körperlänge..µm., 
        Testdf$Paired_data.Bodylength.mm.,
        ylab = "body length [mm]",
        frame = FALSE, names = c("manual", "automatic"), top = TRUE)

boxplot2(Paired_data$Spinalänge..µm., 
         Paired_data$Spinalength.mm.,
         ylab = "body length [mm]",
         frame = FALSE, names = c("manual", "automatic"), top = TRUE)
