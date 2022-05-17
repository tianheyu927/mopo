import os
import numpy as np

ARRAYS_TO_MERGE = [
    "/home/ajc348/rds/hpc-work/softlearning/gym/HalfCheetah/v3/2022-05-16T12-29-56-my-sac-experiment-1/id=8592b_00000-seed=4378/rollouts/rollout_1000000_0.npy"
]
OUTPUT_DIR = "/home/ajc348/rds/hpc-work/mopo/rollouts/softlearning/HalfCheetah/v3/2022-05-16T12-29-56-my-sac-experiment-1/id=8592b_00000-seed=4378"

def main():
    # First ensure that all arrays do exist
    for arr_path in ARRAYS_TO_MERGE:
        assert os.path.exists(arr_path)

    # Also ensure the output directory does not already exist
    if os.path.exists(OUTPUT_DIR):
        raise FileExistsError('Please delete output directory before re-running')
    os.makedirs(OUTPUT_DIR)

    # Loop through the arrays to be combined
    # Add a column which indicates the policy the data came from
    for i, arr_path in enumerate(ARRAYS_TO_MERGE):
        # Load data
        trans_arr = np.load(arr_path)

        # The first array dictates the number of columns that should be present
        if i == 0:
            cols = trans_arr.shape[1]
        assert trans_arr.shape[1] == cols

        # Add a column to the array with the ID of the trajectory
        policy_id = np.full((trans_arr.shape[0], 1), i)
        
        # TODO: Remove this - it is solely for testing purposes
        policy_id[int(trans_arr.shape[0]/2):] = 1

        trans_arr = np.hstack((trans_arr, policy_id))

        # Combine the trajectories
        if i == 0:
            final_arr = np.copy(trans_arr)
        else:
            final_arr = np.vstack((final_arr, np.copy(trans_arr)))

    np.save(os.path.join(OUTPUT_DIR, f'combined_transitions.npy'), final_arr)

if __name__ == "__main__":
    main()
