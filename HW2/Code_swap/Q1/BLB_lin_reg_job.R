# STA 205 - hw2: BLB 
# Author: Roger

mini <- FALSE

#============================== Setup for running on Gauss... ==============================#

args <- commandArgs(TRUE)

cat("Command-line arguments:\n")
print(args)

####
# sim_start ==> Lowest possible dataset number
###

###################
sim_start <- 1000
###################

if (length(args)==0){
  sim_num <- sim_start + 1
  set.seed(121231)
} else {
  # SLURM can use either 0- or 1-indexing...
  # Lets use 1-indexing here...
  sim_num <- sim_start + as.numeric(args[1])
  sim_seed <- (762*(sim_num-1) + 121231)
}

# Find r and s indices:
if(as.numeric(args[1])%%50==0) {
  s_index<-as.numeric(args[1])/50
  r_index<-50
}
if(as.numeric(args[1])%%50!=0) {
  s_index<-ceiling(as.numeric(args[1])/50)
  r_index<-as.numeric(args[1])-(s_index-1)*50
}

cat(paste("\nAnalyzing dataset number ",sim_num,"...\n\n",sep=""))
#============================== Run the simulation study ==============================#

# Load packages:
library(BH)
library(bigmemory.sri)
library(bigmemory)
library(biganalytics)

# I/O specifications:
datapath <- "/home/pdbaines/data"
outpath <- "output/"

# mini or full?
if (mini){
  rootfilename <- "blb_lin_reg_mini"
} else {
  rootfilename <- "blb_lin_reg_data"
}

# Filenames:
input_filename <- paste0(rootfilename,".desc")

# Set up I/O stuff:
input_file <- paste(datapath,input_filename,sep="/")

# Attach big.matrix :
input_data <- attach.big.matrix(dget(input_file),backingpath=datapath)

# Remaining BLB specs:
gamma<-0.7
n<-nrow(input_data)
b<-round(n^gamma)


# Extract the subset:
set.seed(s_index)
index<-sample(1:n,b,replace=TRUE)
subsample<-data.frame(input_data[index,])

# Reset simulation seed:
set.seed(sim_num)

# Bootstrap dataset:
boot_weights<-rmultinom(1,n,prob=rep(1/b,b))

# Fit lm:
beta <- lm(X1001~.-1,data=subsample,weights=boot_weights)$coef

# Output file:
outfile=paste0("output/","coef_",sprintf("%02d",s_index),"_",sprintf("%02d",r_index),".txt")

# Save estimates to file:
write.table(beta, file=outfile,  col.names=TRUE, row.names=FALSE)