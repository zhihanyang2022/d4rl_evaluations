echo "================================================================================"
echo "Running CQL on the D4RL halfcheetah-v1 task ..."

echo "Should I run expert for you, along with other qualities? (y or n)"
# shellcheck disable=SC2162
read RUN_EXPERT

if [ "$RUN_EXPERT" == "y" ]; then
    declare -a QUALITIES=("random" "medium" "expert" "medium-replay" "medium-expert")
else
    declare -a QUALITIES=("random" "medium" "medium-replay" "medium-expert")
fi

for Q in "${QUALITIES[@]}"; do
    echo "================================================================================"
    echo "Running halfcheetah-${Q}-v1 ..."
    python cql/d4rl/examples/cql_mujoco_new.py \
    --env=halfcheetah-"${Q}"-v1 \
    --min_q_weight=5.0 \
    --lagrange_thresh=-1.0 \
    --policy_lr=1e-4 \
    --seed=0 \
    --min_q_version=3
done
