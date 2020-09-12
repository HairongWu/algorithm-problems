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
Use the diagnal vertexes to determine the rectangle
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

### Range Sum of BST

Given the `root` node of a binary search tree, return the sum of values of all nodes with value between `L` and `R` (inclusive).
The binary search tree is guaranteed to have unique values.

Example 1:
```
Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32
```
Example 2:
```
Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
Output: 23
```
Note:

1. The number of nodes in the tree is at most `10000`.
2. The final answer is guaranteed to be less than `2^31`.

Solution:
```
    int rangeSumBST(TreeNode* root, int L, int R) {
        if (!root) return 0;
        if (root->val < L) return rangeSumBST(root->right, L, R);
        if (root->val > R) return rangeSumBST(root->left, L, R);
        return root->val + rangeSumBST(root->left, L, R) + rangeSumBST(root->right, L, R);
    }
 ```
 
 ### Reorder Data in Log Files
 
 You have an array of `logs`.  Each log is a space delimited string of words.
For each log, the first word in each log is an alphanumeric identifier.  Then, either:

- Each word after the identifier will consist only of lowercase letters, or;
- Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

Return the final order of the logs.

Example 1:
```
Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
```
Constraints:

1. `0 <= logs.length <= 100`
2. `3 <= logs[i].length <= 100`
3. `logs[i]` is guaranteed to have an identifier, and a word after the identifier.

Solution:

```
    vector<string> reorderLogFiles(vector<string>& logs) {
        vector<string> res, digitLogs;
        vector<vector<string>> data;
        for (string log : logs) {
            auto pos = log.find(" ");
            if (log[pos + 1] >= '0' && log[pos + 1] <= '9') {
                digitLogs.push_back(log);
                continue;
            }
            data.push_back({log.substr(0, pos), log.substr(pos + 1)});
        }
        sort(data.begin(), data.end(), [](vector<string>& a, vector<string>& b) {
            return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]);
        });
        for (auto &a : data) {
            res.push_back(a[0] + " " + a[1]);
        }
        for (string log : digitLogs) res.push_back(log);
        return res;
    }
```
