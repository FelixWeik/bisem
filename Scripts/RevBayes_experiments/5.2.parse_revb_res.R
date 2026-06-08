library(phangorn)

data_dir = "path_to_revbayes_output"
output_dir = "path_to_summary_file"


n_sims = 200

setwd(output_dir)

res = NULL
for (sim_i in 0:(n_sims - 1)){
    ali_name = paste0(data_dir, "/ali", sim_i)
    true_tree_f = paste0(ali_name, "_true.tre") # true tree
    est_tree_G_f = paste0(ali_name, "_G.tre") # tree estimated under Gamma model
    est_tree_DL_f = paste0(ali_name, "_DL.tre" ) # tree estimated under phyRNN model

    t1 = read.tree( true_tree_f )
    t2 = read.nexus( est_tree_G_f )
    t3 = read.nexus( est_tree_DL_f )
    true_tl = log10(sum(t1$edge.length))

    RF_G = RF.dist(t1, t2, check.labels = TRUE, rooted = FALSE, normalize=TRUE)
    wRF_G = wRF.dist(t1, t2, check.labels = TRUE, rooted = FALSE)
    dTL_G = (true_tl - log10(sum(t2$edge.length)))^2

    RF_DL = RF.dist(t1, t3, check.labels = TRUE, rooted = FALSE, normalize=TRUE)
    wRF_DL = wRF.dist(t1, t3, check.labels = TRUE, rooted = FALSE)
    dTL_DL = (true_tl - log10(sum(t3$edge.length)))^2
    
    res = rbind(res, c(paste0("ali", sim_i), true_tl , RF_G, wRF_G, dTL_G, RF_DL, wRF_DL, dTL_DL))
    
}
colnames(res) = c("sim","true_tl",  "RF_G", "wRF_G", "dTL_G", "RF_DL", "wRF_DL", "dTL_DL")
res = data.frame(res)
res_sim = read.table("summary_res.txt", h=TRUE)
all_res = merge(res, res_sim)

write.table(all_res, file="summary_comb_res.txt", 
            quote=FALSE, row.names=FALSE, sep="\t")




### plot RESULTS
res = read.table("summary_comb_res.txt", h=TRUE)


pdf("summary_res.pdf", height=8, width=14)
par(mfcol=c(2,3))

cols =c("#e41a1c", "#e41a1c","#377eb8", "#4daf4a", "#4daf4a", "#984ea3", "#984ea3","#ff7f00", "#ff7f00")

par(mar = c(12, 2, 2, 2))
boxplot(-res$dLogLik ~ res$model, 
        ylim=c(min(c(0, min(-res$dLogLik))), max(-res$dLogLik)),
        ylab="delta logLik", main="Mean sampled likelihood",
        xlab=NULL,
        col=cols, las=2, lty=1)
abline(h=0, lty=2)

plot(-res$dLogLik ~ log10(res$tree_length), 
     col=cols, 
     pch=16, 
     ylim=c(min(c(0, min(-res$dLogLik))), max(-res$dLogLik)),
     ylab="delta logLik", xlab="Tree length (log10)",
     main="Mean sampled likelihood")

legend("topleft",
       legend = levels(factor(res$model)),
       pch = 16,
       col = cols)

abline(h=0, lty=2)


delta_RF = res$RF_G - res$RF_DL
par(mar = c(12, 2, 2, 2))
boxplot(delta_RF ~ res$model, 
    ylab="delta RF distance", main="Normalized R-F distance", xlab=NULL,
    col=cols, las=2, lty=1)
abline(h=0, lty=2)

delta_wRF = res$wRF_G - res$wRF_DL
par(mar = c(12, 2, 2, 2))
boxplot(delta_wRF ~ res$model, 
    ylab="delta WRF distance", main="Weighted R-F distance", xlab=NULL,
    col=cols, las=2, lty=1)
abline(h=0, lty=2)



# plot TL
delta_tlSE = log10(res$dTL_G / res$dTL_DL)
par(mar = c(12, 2, 2, 2))
boxplot(delta_tlSE ~ res$model, 
    ylab="log10 SE ratio", main="Tree length accuracy", xlab=NULL,
    col=cols, las = 2, lty=1)
abline(h=0, lty=2)

plot(delta_tlSE ~ log10(res$tree_length), 
     col=cols, 
     pch=16, 
     ylab="log10 SE ratio", xlab="True tree length (log10)",
     main="Tree length accuracy")

abline(h=0, lty=2)
dev.off()

#### RES FIGURE
outline = FALSE
library(scales)
pdf("summary_boxplots.pdf", height=9, width=4)
par(mfcol=c(5,1))
al = 0.5
cols =c(alpha("#e41a1c", al), "#e41a1c","#377eb8", alpha("#4daf4a", al), "#4daf4a", alpha("#984ea3", al), "#984ea3",alpha("#ff7f00", al), "#ff7f00")

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





#################
# Calculate ESS values
library(coda)
n_sims = 200

get_ess <- function(wd, burnin = 500, var="Likelihood"){ # Posterior
    res = NULL
    for (sim_i in 0:(n_sims - 1)){
        ali_name = paste0(wd, "/ali", sim_i)
        log_G_f = read.table(paste0(ali_name, "_G.log"), h=T)
        log_DL_f = read.table(paste0(ali_name, "_DL.log"), h=T)
    
        post_G = as.numeric(unlist(log_G_f[var]))
        post_DL = as.numeric(unlist(log_DL_f[var]))
 
        essG = effectiveSize(post_G[burnin:dim(log_G_f)[1]])
        essDL = effectiveSize(post_DL[burnin:dim(log_DL_f)[1]])
    
        res = rbind(res, c(paste0("ali", sim_i), essG, essDL))
    
    }
    colnames(res) = c("sim", "ess_G", "ess_DL")
    res = data.frame(res)
}

res = get_ess("data_dir", burnin=500)
