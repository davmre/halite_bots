import requests
import json
import os

# outputs a list of filenames corresponding to replays from all current Gold-tier players on the Halite leaderboard. 

def get_top_users():
    # scrapes the leaderboard for users in the given tiers, returns a
    # list of (username, numeric userid) tuples

    n_to_fetch = 1
    tiers=("Gold", "Diamond")
    
    result = []
    
    base_url = "https://halite.io/api/web/user?fields[]=isRunning&values[]=1&orderBy=rank&limit=%d&page=0" % n_to_fetch
    r = requests.get(base_url)
    user_list = json.loads(r.content)["users"]
    for user in user_list:
        if user["tier"] in tiers:
            rtuple = (user["username"], int(user["userID"]))
            result.append(rtuple)

    return result
    
def get_replay_fnames(user_id, n_to_fetch=1000):
    url = "https://halite.io/api/web/game?userID=%d&limit=%d" % (user_id, n_to_fetch)
    r = requests.get(url)
    game_list = json.loads(r.content)
    return [game["replayName"] for game in game_list]
    
def main():
    users = get_top_users()
    print users

    all_replays = []
    for username, user_id in users:
        replay_fnames = get_replay_fnames(user_id)
        for fname in replay_fnames:
            all_replays.append((username, fname))
            print username, fname

        
if __name__ == "__main__":
    main()

    
