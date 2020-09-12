# algorithm-problems
A collection of algorithmic problems

## LeetCode

### Distinct Subsequences II

Given a string `S`, count the number of distinct, non-empty subsequences of `S`.

Since the result may be large, return the answer modulo `10^9 + 7`.

Example 1:
```
Input: "abc"
Output: 7
Explanation: The 7 distinct subsequences are "a", "b", "c", "ab", "ac", "bc", and "abc".
```

Example 2:
```
Input: "aba"
Output: 6 Explanation: The 6 distinct subsequences are "a", "b", "ab", "ba", "aa" and "aba".
```

Example 3:
```
Input: "aaa"
Output: 3 Explanation: The 3 distinct subsequences are "a", "aa" and "aaa".
```

Note:

1. `S` contains only lowercase letters.
2. `1 <= S.length <= 2000`

Solution:

Use the state transition equation of dynamic programming
```
int distinctSubseqII(string S) {
    int M = 1e9 + 7;
    vector<int> dp(26); //there are 26 different lowercase charactors
    for (char c : S) {
        dp[c - 'a'] = accumulate(dp.begin(), dp.end(), 1L) % M; //Computes the sum of the given value init and the elements in the range [first, last). 
    }
    return accumulate(dp.begin(), dp.end(), 0L) % M;
}
```

### Minimum Area Rectangle

Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points, with sides parallel to the x and y axes.
If there isn't any rectangle, return 0.

Example 1:
```
Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]
Output: 4
```
Example 2:
```
Input: [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]
Output: 2
```
Note:

1. `1 <= points.length <= 500`
2. `0 <= points[i][0] <= 40000`
3. `0 <= points[i][1] <= 40000`
4. All points are distinct.

Solution:
```
    int minAreaRect(vector<vector<int>>& points) {
        int res = INT_MAX, n = points.size();
        unordered_map<int, unordered_set<int>> m;
        for (auto point : points) {
            m[point[0]].insert(point[1]);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (points[i][0] == points[j][0] || points[i][1] == points[j][1]) continue;
                if (m[points[i][0]].count(points[j][1]) && m[points[j][0]].count(points[i][1])) {
                    res = min(res, abs(points[i][0] - points[j][0]) * abs(points[i][1] - points[j][1]));
                }   
            }
        }
        return res == INT_MAX ? 0 : res;
    }
```
