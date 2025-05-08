# for(auto i:matrix) 和 for(const auto& i:matrix)

[原题链接](https://leetcode.cn/problems/search-a-2d-matrix-ii/?envType=study-plan-v2&envId=top-100-liked)

```c++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int n=matrix.size(),m=matrix[0].size();
        for(auto i:matrix) // [!code --]
        for(auto& i:matrix) // [!code ++]
        {
            auto it = lower_bound(i.begin(),i.end(),target);
            if(it!=i.end()&&*it==target) return true;
        }
        return false;
    }
};
```

在 C++ 中，for(auto i:matrix) 和 for(const auto& i:matrix) 的区别在于：

* for(auto i:matrix) 会拷贝每个子数组（`vector<int>`），这会消耗额外的内存和时间，特别是当矩阵很大时。
* for(auto& i:matrix) 是引用每个子数组，不会拷贝数据，因此更高效。
  