#!/bin/bash
# One-shot watcher (IR2/ZC1 rebuild): the detached driver launched via zsh `&` was niced
# (zsh BG_NICE) onto efficiency cores. Wait for stage 1's sims to finish writing (the LAST
# file gen_simulations saves is sample_mc_correlations.npy), kill the niced tree, and
# relaunch the resumable driver at inherited nice 0 (nohup -> survives harness death).
cd "$(dirname "$0")"
OUT=../outputs/simulations
want=4
echo "watcher: waiting for $want completed sim sets ($(date))"
while :; do
  n=0
  for SIM in sim_z1 sim_z1_high_mc_scatter sim_z1_high_noise sim_z1_high_rm_scatter; do
    [ -f "$OUT/${SIM}.10000.376.delta_sigma.corrected_mean/sample_mc_correlations.npy" ] && n=$((n+1))
  done
  [ "$n" -ge "$want" ] && break
  sleep 60
done
echo "watcher: all $want sim sets complete ($(date)); grace 30s for final flush"
sleep 30
echo "watcher: killing niced driver tree"
pkill -f run_paper1_cmean_par.sh 2>/dev/null
sleep 2
pkill -f "train_inferrer.py .*corrected_mean" 2>/dev/null
pkill -f "caffeinate -i bash run_paper1_cmean_par.sh" 2>/dev/null
sleep 3
echo "watcher: relaunching driver at nice $(nice) ($(date))"
nohup caffeinate -i bash run_paper1_cmean_par.sh >> /tmp/cmean_production.log 2>&1 &
echo "watcher: relaunched pid $! (nice $(nice)); done"
