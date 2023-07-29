#!/usr/bin/env Rscript --vanilla

# Usage: Rscript --vanilla scripts/plot_dot_plots.r \
#    "data/combined_data.20220520_subset.metadata/participant_specimen_disease_map.tsv" \
#    "data/combined_data.20220520_subset.dotplots/input" \
#    "data/combined_data.20220520_subset.dotplots/output";
# Fill in the above using config.paths.dataset_specific_metadata, config.paths.dotplots_input, and config.paths.dotplots_output.

args = commandArgs(trailingOnly=TRUE)

if (length(args) < 3) {
  stop("Three arguments must be supplied (specimen metadata file, data directory, output directory).n", call.=FALSE)
}

fname_specimen_metadata = args[1]
data_dir = args[2]
output_dir = args[3]

library(ggplot2)
library(tidyr)
library(dplyr)
# library(reshape2)
library(stringr)

library(grid)
library(gridExtra)
# library(ggpubr)

specimen_metadata = read.table(fname_specimen_metadata, header=T, sep="\t")
# head(specimen_metadata)
# head(specimen_metadata$specimen_time_point)

# Replace empty string with N/A
specimen_metadata$specimen_time_point=na_if(specimen_metadata$specimen_time_point, "")
# head(specimen_metadata$specimen_time_point)

# Fill N/A with specimen label: If specimen_time_point is N/A or empty string, use specimen_label instead as the timepoint
specimen_metadata$Timepoint = coalesce(specimen_metadata$specimen_time_point, specimen_metadata$specimen_label)
# head(specimen_metadata$Timepoint)
# head(specimen_metadata)

plot_dotplot_for_participant <- function(participant, fname_data, specimen_metadata) {
    isotype.order <- c('IGHM','IGHD','IGHG3','IGHG1','IGHA1','IGHG2','IGHG4','IGHE','IGHA2')

    data <- read.table(fname_data, header=T, sep="\t")
    # head(data)

    # Merge
    data.anno = left_join(data, specimen_metadata, by=c("participant_label", "specimen_label"))
    # head(data.anno)

    stopifnot(dim(data.anno)[1] == dim(data)[1])
    stopifnot(!any(is.na(data.anno$Timepoint)))

    timepoint_order = distinct(data.anno, Timepoint)$Timepoint # get unique
    timepoint_order = timepoint_order[order(timepoint_order)] # sort
    # timepoint_order

    #get the subject and timepoint
    data.subset <- subset(data.anno,participant_label == participant)

    ##ordering for the timepoints
    data.subset$Timepoint <- factor(data.anno$Timepoint,timepoint_order);


    #add bins for the size scale for dot plots
    # clone size = reads per clone
    data.subset$cutSize <- cut(data.subset$clone_size, breaks = c(1,10,100,500,1000,5000,Inf),right = FALSE)

    data.subset$extracted_isotype <- factor(data.subset$extracted_isotype,isotype.order);
    col.scale.cols.alt <- c("green", "white", "magenta", "magenta1", "magenta2", "magenta3", "magenta4")

    #draw the plot
    data.subset.plots <-
      ggplot(data.subset, aes(x=v_gene, y=cdr3_aa_sequence_trim_len)) +
      geom_point(aes(fill=clone_v_mut_median, size=cutSize), position=position_jitter(w=0.4, h=0.4), shape=21, stroke=0.1) +
      scale_y_continuous(breaks=seq(5,40,5)) +
      scale_fill_gradientn(colours=col.scale.cols.alt, name="V-REGION\nmutation\n(%)") +
      scale_size_manual(
        breaks = c("[1,10)", "[10,100)", "[100,500)", "[500,1e+03)", "[1e+03,5e+03)", "[5e+03,Inf)"),
        limits =  c("[1,10)", "[10,100)", "[100,500)", "[500,1e+03)", "[1e+03,5e+03)", "[5e+03,Inf)"),
        values = c(1,2,3,4,5,6),
        labels = c("<10", "<100", "<500", "<1000", "<5000", ">5000"),
        name = "Read\nCount") +
      labs(x="IGHV genes", y="CDR3 length (aa)") +
      theme(axis.text.x  = element_blank(),
            axis.text.y = element_text(size=18),
            axis.ticks.x = element_blank(),
            axis.ticks.y = element_blank(),
            legend.position="right",
            strip.background = element_blank(),
            strip.text.x = element_text(size=18),
            strip.text.y = element_text(size=18),
            #strip.background = element_rect(colour=SeroTest),
            legend.key = element_blank()) +
      theme(plot.background = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major = element_blank(),
            panel.background = element_rect(fill="black")) +
      #scale_y_continuous(breaks=seq(0,30,3)) +
      facet_grid(Timepoint~extracted_isotype,drop=FALSE)

    return(data.subset.plots)
}

for(participant in unique(specimen_metadata$participant_label)) {
    fname_in = paste0(data_dir, "/", participant, ".tsv")
    if(!file.exists(fname_in)) {
        message(paste("Data not found for:", participant))
        next
    }
    print(participant)
    plot = plot_dotplot_for_participant(participant, fname_in, specimen_metadata)
    ggsave(filename=paste0(output_dir, '/', participant, ".dotplots.png"), plot, width=11, height=19,dpi=600,limitsize = FALSE)
    ggsave(filename=paste0(output_dir, '/', participant, ".dotplots.pdf"), plot, width=11, height=19,dpi=600,limitsize = FALSE)
}
