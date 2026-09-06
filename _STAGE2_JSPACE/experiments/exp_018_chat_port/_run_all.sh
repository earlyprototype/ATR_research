#!/bin/sh
# The commands the registered run used. `set -e` was added after the run, in
# the review fixes: the registered run used these same two commands without it.
#
# Rerunning these two commands against the committed results files does
# nothing, because both arms are complete: the resume finds every prompt
# already finished and exits without writing. If a prompt were missing, the
# resume would stop, because neither committed results file records the weights
# revision it was run on and the runner will not run new prompts on weights it
# cannot match to the old ones. To finish such an arm, add
# --assume-legacy-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, which is
# this run's revision inferred from this machine's cache holding exactly one
# version of the weights, and which the results file then records as assumed
# rather than measured.
set -eux
cd "$(dirname "$0")"
python3 run_exp018.py --stage loop --arm bare --dtype bfloat16 --max-iter 150 --check-start 10 --check-every 2 --seed 42 --resume >> output/loop_bare.log 2>&1
python3 run_exp018.py --stage loop --arm chat --dtype bfloat16 --max-iter 150 --check-start 10 --check-every 2 --seed 42 --resume >> output/loop_chat.log 2>&1
echo "ALL LOOPS DONE" >> output/loop_chat.log
