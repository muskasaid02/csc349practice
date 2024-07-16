def find_max_weight_activities(activities):
    # Sort activities by start time
    activities.sort(key=lambda x: x[0])

    n = len(activities)
    dp = [0] * (n + 1)  # dp[i] will store the maximum weight of non-conflicting activities up to i
    activities.insert(0, (0, 0, 0))  # Add a dummy activity for easier indexing

    # Function to find the latest non-conflicting activity
    def latest_non_conflict(j):
        for i in range(j - 1, 0, -1):
            if activities[i][1] <= activities[j][0]:
                return i
        return 0

    # Base case
    dp[0] = 0  # With zero activities, the maximum weight is 0

    # Fill the DP table using the recursive formula
    for j in range(1, n + 1):
        incl_weight = activities[j][2] + dp[latest_non_conflict(j)]
        excl_weight = dp[j-1]
        dp[j] = max(incl_weight, excl_weight)

    # The cell that holds the solution
    return dp[n]

# Example usage:
# activities is a list of tuples (start_time, finish_time, weight)
activities = [(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4), (5, 8, 11), (7, 9, 2)]
max_weight = find_max_weight_activities(activities)
print(f"The maximum weight of a set of non-conflicting activities is {max_weight}")



def max_contiguous_subsequence_sum(arr):
    if not arr:
        return 0

    n = len(arr)
    dp = [0] * n

    # Base case
    dp[0] = arr[0]
    
    for i in range(1, n):
        dp[i] = max(arr[i], arr[i] + dp[i-1])

    # The cell that holds the solution
    return max(dp)

# Example usage:
arr = [5, 15, -30, 10, -5, 40, 10]
max_sum = max_contiguous_subsequence_sum(arr)
print(f"The maximum sum of a contiguous subsequence is {max_sum}")




def min_penalty(hotels):
    # Number of hotels
    n = len(hotels)
    
    # Initialize DP array
    dp = [float('inf')] * (n + 1)
    
    # Starting point penalty is 0
    dp[0] = 0
    hotels.insert(0, 0)  # Insert the starting point at position 0
    
    # Fill the DP table using the recursive formula
    for i in range(1, n + 1):
        for j in range(i):
            dp[i] = min(dp[i], dp[j] + (300 - (hotels[i] - hotels[j]))**2)
    
    # The cell that holds the solution
    return dp[n]

# Example usage:
hotels = [100, 200, 400, 600, 700, 900]  # Example hotel mileposts
min_total_penalty = min_penalty(hotels)
print(f"The minimum total penalty for the trip is {min_total_penalty}")



def max_total_profit(locations, profits, k):
    n = len(locations)
    dp = [0] * (n + 1)
    
    # Preprocess: remove locations where distance < k
    filtered_locations = [(locations[i], profits[i]) for i in range(n) if locations[i] >= k]
    if not filtered_locations:
        return 0
    
    locations, profits = zip(*filtered_locations)
    n = len(locations)
    
    # Initialize DP array
    dp = [0] * (n + 1)
    
    # Fill the DP table using the recursive formula
    for i in range(1, n + 1):
        max_profit = 0
        for j in range(i):
            if locations[i - 1] - locations[j - 1] > k:
                max_profit = max(max_profit, dp[j])
        dp[i] = profits[i - 1] + max_profit
    
    # The cell that holds the solution
    return max(dp)

# Example usage:
locations = [2, 5, 9, 14, 19]
profits = [4, 2, 7, 3, 8]
k = 5
max_profit = max_total_profit(locations, profits, k)
print(f"The maximum expected total profit is {max_profit}")




def longest_common_substring(x, y):
    n = len(x)
    m = len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_length = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0

    return max_length

# Example usage:
x = "dcabac"
y = "cbcbabca"
max_length = longest_common_substring(x, y)
print(f"The length of the longest common substring is {max_length}")



def gene_alignment(x, y, delta):
    n = len(x)
    m = len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Initialize base cases
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + delta[x[i-1]]['-']
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + delta['-'][y[j-1]]

    # Fill the DP table using the recursive formula
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i-1][j-1] + delta[x[i-1]][y[j-1]],
                dp[i-1][j] + delta[x[i-1]]['-'],
                dp[i][j-1] + delta['-'][y[j-1]]
            )

    return dp[n][m]

# Example usage:
x = "ATGCC"
y = "TACGCA"
delta = {
    'A': {'A': 1, 'C': -1, 'G': -1, 'T': -1, '-': -2},
    'C': {'A': -1, 'C': 1, 'G': -1, 'T': -1, '-': -2},
    'G': {'A': -1, 'C': -1, 'G': 1, 'T': -1, '-': -2},
    'T': {'A': -1, 'C': -1, 'G': -1, 'T': 1, '-': -2},
    '-': {'A': -2, 'C': -2, 'G': -2, 'T': -2, '-': 0}
}
max_score = gene_alignment(x, y, delta)
print(f"The highest scoring alignment has a score of {max_score}")









def max_profit_cloth_cutting(X, Y, products):
    # Initialize the DP table
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    
    # Fill the DP table using the recursive formula
    for w in range(1, X + 1):
        for h in range(1, Y + 1):
            for product in products:
                a, b, c = product
                if a <= w and b <= h:
                    dp[w][h] = max(dp[w][h], c + dp[w - a][h - b])
            for k in range(1, w):
                dp[w][h] = max(dp[w][h], dp[k][h] + dp[w - k][h])
            for k in range(1, h):
                dp[w][h] = max(dp[w][h], dp[w][k] + dp[w][h - k])
    
    return dp[X][Y]

# Example usage:
# X, Y are the dimensions of the cloth
# products is a list of tuples (a_i, b_i, c_i) where a_i and b_i are the 
# dimensions needed for product i and c_i is the selling price
X = 4
Y = 5
products = [(1, 2, 3), (2, 2, 4), (3, 1, 5)]
max_profit = max_profit_cloth_cutting(X, Y, products)
print(f"The maximum profit obtainable is {max_profit}")


def max_profit_cloth_cutting(X, Y, products):
    # Initialize the DP table
    dp = [[[0] * (len(products) + 1) for _ in range(Y + 1)] for _ in range(X + 1)]
    
    # Fill the DP table using the recursive formula
    for i in range(1, len(products) + 1):
        a_i, b_i, c_i = products[i-1]
        for x in range(1, X + 1):
            for y in range(1, Y + 1):
                # Don't include the current product
                dp[x][y][i] = dp[x][y][i-1]
                
                # Include the current product if it fits horizontally
                if a_i <= x and b_i <= y:
                    dp[x][y][i] = max(dp[x][y][i], c_i + dp[x - a_i][y][i] + dp[a_i][y - b_i][i])
                    
                # Include the current product if it fits vertically
                if b_i <= x and a_i <= y:
                    dp[x][y][i] = max(dp[x][y][i], c_i + dp[x][y - b_i][i] + dp[x - b_i][a_i][i])
                    
                # Horizontal cuts
                for k in range(1, x):
                    dp[x][y][i] = max(dp[x][y][i], dp[k][y][i] + dp[x - k][y][i])
                
                # Vertical cuts
                for k in range(1, y):
                    dp[x][y][i] = max(dp[x][y][i], dp[x][k][i] + dp[x][y - k][i])
    
    return dp[X][Y][len(products)]

# Example usage:
# X, Y are the dimensions of the cloth
# products is a list of tuples (a_i, b_i, c_i) where a_i and b_i are the dimensions needed for product i and c_i is the selling price
X = 4
Y = 5
products = [(1, 2, 3), (2, 2, 4), (3, 1, 5)]
max_profit = max_profit_cloth_cutting(X, Y, products)
print(f"The maximum profit obtainable is {max_profit}")



def can_make_change(coins, V):
    # Initialize the DP table
    dp = [0] * (V + 1)
    
    # Base case
    dp[0] = 1
    
    # Fill the DP table using the recursive formula
    for i in range(1, V + 1):
        for coin in coins:
            if i >= coin:
                dp[i] = max(dp[i], dp[i - coin])
    
    return dp[V] == 1

# Example usage:
# coins is a list of coin denominations
# V is the value to make change for
coins = [1, 3, 4]
V = 6
is_possible = can_make_change(coins, V)
print(f"Is it possible to make change for {V} using the coins {coins}? {'Yes' if is_possible else 'No'}")


def can_make_change_once(coins, V):
    n = len(coins)
    dp = [[0] * (V + 1) for _ in range(n + 1)]

    # Base case: we can always make change for 0 value with 0 coins
    for i in range(n + 1):
        dp[i][0] = 1

    # Fill the DP table using the recursive formula
    for i in range(1, n + 1):
        for v in range(1, V + 1):
            dp[i][v] = dp[i-1][v]  # not using the i-th coin
            if v >= coins[i-1]:
                dp[i][v] = max(dp[i][v], dp[i-1][v - coins[i-1]])  # using the i-th coin if possible

    return dp[n][V] == 1

# Example usage:
# coins is a list of coin denominations
# V is the value to make change for
coins = [1, 5, 10, 20]
V = 31
is_possible = can_make_change_once(coins, V)
print(f"Is it possible to make change for {V} using the coins {coins} at most once? {'Yes' if is_possible else 'No'}")




