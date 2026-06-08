library(vioplot)
library(phangorn)

data_dir = "path_to_revbayes_output"
output_dir = "path_to_summary_file"


n_sims = 100
tag =  "50d" #

setwd(output_dir)

res = NULL
for (sim_i in 0:(n_sims-1)){
    ali_name = paste0(data_dir, "/ali", sim_i)
    true_tree_f = paste0(ali_name, "_true.tre")
    est_tree_G_f = paste0(ali_name, "_G_MAP.tre")
    est_tree_DL_f = paste0(ali_name, "_DL", tag, "_MAP.tre" )

    t1 = read.tree( true_tree_f )
    t2 = read.nexus( est_tree_G_f )
    t3 = read.nexus( est_tree_DL_f )
    true_tl = log10(sum(t1$edge.length))

    RF_G = RF.dist(t1, t2, check.labels = TRUE, rooted = FALSE, normalize=TRUE)
    wRF_G = wRF.dist(t1, t2, check.labels = TRUE, rooted = FALSE, normalize=TRUE)
    dTL_G = (true_tl - log10(sum(t2$edge.length)))^2

    RF_DL = RF.dist(t1, t3, check.labels = TRUE, rooted = FALSE, normalize=TRUE)
    wRF_DL = wRF.dist(t1, t3, check.labels = TRUE, rooted = FALSE, normalize=TRUE)
    dTL_DL = (true_tl - log10(sum(t3$edge.length)))^2
    
    res = rbind(res, c(paste0("ali", sim_i), true_tl , RF_G, wRF_G, dTL_G, RF_DL, wRF_DL, dTL_DL))
    
}
colnames(res) = c("sim","true_tl",  "RF_G", "wRF_G", "dTL_G", "RF_DL", "wRF_DL", "dTL_DL")
res = data.frame(res)

res_sim = read.table(paste0("summary_res_", tag, ".txt"), h=TRUE)
all_res = merge(res, res_sim)

write.table(all_res, file=paste0("summary_comb_", tag, "res.txt"), 
            quote=FALSE, row.names=FALSE, sep="\t")




res = read.table(paste0("summary_comb_", tag, "res.txt"), h=TRUE)



sum(res$wRF_G > res$wRF_DL) / length(res$wRF_DL)
sum(res$wRF_G < res$wRF_DL) / length(res$wRF_DL)

sum(res$RF_G > res$RF_DL) / length(res$wRF_DL)
sum(res$RF_G < res$RF_DL) / length(res$wRF_DL)

sum(res$dTL_G > res$dTL_DL) / length(res$wRF_DL)
sum(res$dTL_G < res$dTL_DL) / length(res$wRF_DL)




#### RES FIGURE
outline = FALSE
library(scales)
pdf(paste0("summary_boxplots_nooutline_", tag, ".pdf"), height=9, width=4)
par(mfcol=c(5,1))
al = 0.5
cols =c(alpha("#e41a1c", al), "#e41a1c",
        "#377eb8", 
        alpha("#4daf4a", al), "#4daf4a", 
        alpha("#984ea3", al), "#984ea3",
        alpha("#ff7f00", al), "#ff7f00")

left_marg = 5
top_marg = 0.5
right_marg=1
bottom_marg = 0
par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(-res$dLogLik ~ res$model, 
        ylim=c(min(c(0, min(-res$dLogLik))), max(-res$dLogLik)),
        ylab="delta log likelihood", #main="Mean sampled likelihood",
        xlab=NULL,outline=outline,
        col=cols, las=2, lty=1, xaxt="n")
abline(h=0, lty=2)


delta_RF = res$RF_G - res$RF_DL
par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(delta_RF ~ res$model, 
    ylab="delta R-F distance", #main="Normalized R-F distance", 
    xlab=NULL,  outline=outline,
    col=cols, las=2, lty=1, xaxt="n")
abline(h=0, lty=2)

delta_wRF = res$wRF_G - res$wRF_DL
par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(delta_wRF ~ res$model, 
    ylab="delta weighted R-F distance", #main="Weighted R-F distance", 
    xlab=NULL,  outline=outline,
    col=cols, las=2, lty=1, xaxt="n")
abline(h=0, lty=2)



# plot TL
delta_tlSE = log10(res$dTL_G / res$dTL_DL)
par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(delta_tlSE ~ res$model, 
    ylab="Tree length SE ratio (log10)", #main="Tree length accuracy", 
    xlab=NULL,    outline=outline,
    col=cols, las = 2, lty=1)
abline(h=0, lty=2)


abline(h=0, lty=2)
dev.off()



#### RES OVERALL FIGURE
tag = "4d"
res = read.table(paste0("summary_comb_", tag, "res.txt"), h=TRUE)
tag = "10d" 
res10 = read.table(paste0("summary_comb_", tag, "res.txt"), h=TRUE)
tag = "20d" 
res20 = read.table(paste0("summary_comb_", tag, "res.txt"), h=TRUE)
tag = "50d" 
res50 = read.table(paste0("summary_comb_", tag, "res.txt"), h=TRUE)

boxplot(cbind(log10(res$dTL_G / res$dTL_DL  ), 
              log10(res$dTL_G / res10$dTL_DL), 
              log10(res$dTL_G / res20$dTL_DL), 
              log10(res$dTL_G / res50$dTL_DL)), outline=F)
boxplot(cbind(res$wRF_G, res$wRF_DL, res10$wRF_DL, res20$wRF_DL), outline=F)


delta_wRF = res$wRF_G - res$wRF_DL
delta_wRF10 = res$wRF_G - res10$wRF_DL
delta_wRF20 = res$wRF_G - res20$wRF_DL

boxplot(cbind(delta_wRF,delta_wRF10,delta_wRF20), outline=FALSE)
abline(h=0, lty=2)



outline = F
library(scales)
# pdf(paste0("summary_boxplots_nooutline_", tag, ".pdf"), height=9, width=4)
# par(mfcol=c(5,1))
par(mfrow=c(1,4))
al = 0.5
cols =c(alpha("#e41a1c", al), "#e41a1c",
        "#377eb8", 
        alpha("#4daf4a", al), "#4daf4a", 
        alpha("#984ea3", al), "#984ea3",
        alpha("#ff7f00", al), "#ff7f00")

left_marg = 5
top_marg = 0.5
right_marg=1
bottom_marg = 1
# par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(cbind(-res$dLogLik, -res10$dLogLik) , 
        ylim=c(min(c(0, min(-res$dLogLik))), max(-res$dLogLik)),
        ylab="delta log likelihood", #main="Mean sampled likelihood",
        xlab=NULL,outline=outline,
        col=cols, las=2, lty=1) #, xaxt="n")
abline(h=0, lty=2)


delta_RF4 = res$RF_G - res$RF_DL
delta_RF10 = res10$RF_G - res10$RF_DL
# par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(cbind(delta_RF4, delta_RF10), 
    ylab="delta R-F distance", #main="Normalized R-F distance", 
    xlab=NULL,  outline=outline,
    col=cols, las=2, lty=1, xaxt="n")
abline(h=0, lty=2)

delta_wRF4 = res$wRF_G - res$wRF_DL
delta_wRF10 = res10$wRF_G - res10$wRF_DL
# par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
boxplot(cbind(delta_wRF4, delta_wRF10),  
    ylab="delta weighted R-F distance", #main="Weighted R-F distance", 
    xlab=NULL,  outline=outline,
    col=cols, las=2, lty=1, xaxt="n")
abline(h=0, lty=2)



# plot TL
delta_tlSE4 = log10(res$dTL_G / res$dTL_DL)
delta_tlSE10 = log10(res10$dTL_G / res10$dTL_DL)
# par(mar = c(bottom_marg, left_marg, top_marg, right_marg))
vioplot(cbind(delta_tlSE4, delta_tlSE10 ),
    ylab="Tree length SE ratio (log10)", #main="Tree length accuracy", 
    xlab=NULL,    outline=outline,
    col=cols, las = 2, lty=1)
abline(h=0, lty=2)


abline(h=0, lty=2)
dev.off()




#################
# CHECK ESS values
library(coda)

get_ess <- function(wd, burnin = 500, var="Posterior"){ # Posterior
    res = NULL
    for (sim_i in 0:(n_sims-1)){
        ali_name = paste0(wd, "/ali", sim_i)
        log_G_f = read.table(paste0(ali_name, "_G.log"), h=T)
        log_DL_f = read.table(paste0(ali_name, "_DL", tag, ".log"), h=T)
    
        post_G = as.numeric(unlist(log_G_f[var]))
        post_DL = as.numeric(unlist(log_DL_f[var]))
 
        essG = effectiveSize(post_G[burnin:dim(log_G_f)[1]])
        essDL = effectiveSize(post_DL[burnin:dim(log_DL_f)[1]])
    
        res = rbind(res, c(paste0("ali", sim_i), essG, essDL))
    
    }
    colnames(res) = c("sim", "ess_G", "ess_DL")
    res = data.frame(res)
}

res50 = get_ess(data_dir, burnin=500)

write.table(res50, file=paste0("ess_res_", tag, ".txt"), 
            quote=FALSE, row.names=FALSE, sep="\t")


res50 = read.table(paste0("ess_res_", tag, ".txt"),h=T, sep="\t")


boxplot(res50[,c(2,3)])

deltaESS50 = res50$ess_G - res50$ess_DL


quantile(-deltaESS50, p=c(0.025, 0.5, 0.975))
length(deltaESS50[deltaESS50 < 0]) / length(deltaESS50)


#################


