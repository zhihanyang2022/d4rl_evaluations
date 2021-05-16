echo "================================================================================"
echo "Running BC on the D4RL halfcheetah-v1 task ..."

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
    python brac/scripts/train_bc.py \
    --env_name=halfcheetah-"${Q}"-v1 \
    --seed=0
done
