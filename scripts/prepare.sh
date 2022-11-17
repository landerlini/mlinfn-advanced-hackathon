#!/bin/bash

args=`echo $@`  # Transform new lines in spaces
users=`python3 -c "print (' '.join([w for w in '$args'.split(' ') if w != '' and w[0] != '-']))"`

## Flags
cleanup=`python3 -c "print ('yes' if '--cleanup' in '$args'.split(' ') else 'no')"`
cleanup_da=`python3 -c "print ('yes' if '--cleanup-da' in '$args'.split(' ') else 'no')"`
cleanup_unet=`python3 -c "print ('yes' if '--cleanup-unet' in '$args'.split(' ') else 'no')"`
cleanup_gnn=`python3 -c "print ('yes' if '--cleanup-gnn' in '$args'.split(' ') else 'no')"`
cleanup_xai=`python3 -c "print ('yes' if '--cleanup-xai' in '$args'.split(' ') else 'no')"`
solutions_da=`python3 -c "print ('yes' if '--solutions-da' in '$args'.split(' ') else 'no')"`
solutions_unet=`python3 -c "print ('yes' if '--solutions-unet' in '$args'.split(' ') else 'no')"`
solutions_gnn=`python3 -c "print ('yes' if '--solutions-gnn' in '$args'.split(' ') else 'no')"`
solutions_xai=`python3 -c "print ('yes' if '--solutions-xai' in '$args'.split(' ') else 'no')"`

echo "Users: $users"

REMOTE=https://github.com/landerlini/mlinfn-advanced-hackathon
LOCAL=$HOME/mlinfnrepo
rm -rf $HOME/mlinfnrepo
git clone $REMOTE $LOCAL/


if [ $solutions_da = 'no' ];
  then
    rm $LOCAL/ex/domain_adaptation/Excercise_DA_MLhackathon.ipynb
    rm $LOCAL/ex/domain_adaptation/Excercise_DA_MLhackathon_SimpleDNN.ipynb
  fi;
  
if [ $solutions_xai = 'no' ];
  then
    rm $LOCAL/ex/xai/XAI-version2.ipynb
  fi;

if [ $solutions_gnn = 'no' ];
  then
    rm $LOCAL/ex/gnn_transformers/TransformerSG_solution.ipynb
    rm $LOCAL/ex/gnn_transformers/GNN_IN_JetTagger/GNN_Jet_Tagging_IN.ipynb
  fi;

if [ $solutions_unet = 'no' ];
  then
    rm -rf $LOCAL/ex/unet/ex_solution
  fi;


for user in $users;
  do
    echo "Processing $user"

    PRIVATE=/jupyter-mounts/users/$user
    SHARED=/jupyter-mounts/shared/$user


    ## Clean selectively to allow refreshing
    if [ $cleanup_unet = 'yes' ]; then rm -rf $SHARED/ex/unet; fi
    if [ $cleanup_da = 'yes' ]; then rm -rf $SHARED/ex/domain_adaptation; fi
    if [ $cleanup_gnn = 'yes' ]; then rm -rf $SHARED/ex/gnn_transformers; fi
    if [ $cleanup_xai = 'yes' ]; then rm -rf $SHARED/ex/xai; fi

    if [ $cleanup = 'yes' ];
      then
        rm -rf $PRIVATE/*-*;
        rm -rf $SHARED/Tuesday;
        rm -rf $SHARED/Wednesday;
      fi


    ## Private
    mkdir -p $PRIVATE
    rsync --ignore-existing -r $LOCAL/advanced_jupyter/*         $PRIVATE/Jupyter-Anderlini/
    rsync --ignore-existing -r $LOCAL/introduction_to_pytorch/*  $PRIVATE/Pytorch-Giagu/
    rsync --ignore-existing -r $LOCAL/introduction_to_gnns/*     $PRIVATE/GNN-Rizzi/

    ## Shared
    mkdir -p $SHARED/Tuesday
    mkdir -p $SHARED/Wednesday

    rsync --ignore-existing -r $LOCAL/ex/domain_adaptation/*     $SHARED/Tuesday/Domain_Adaptation_in_HEP/
    rsync --ignore-existing -r $LOCAL/ex/unet/*                  $SHARED/Tuesday/Lung_Segmentation_with_UNets/
    rsync --ignore-existing -r $LOCAL/ex/gnn_transformers/*      $SHARED/Wednesday/GNNs_and_Transformers/
    rsync --ignore-existing -r $LOCAL/ex/xai/*                   $SHARED/Wednesday/Explainable_AI
  done;
