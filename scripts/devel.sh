repo=${repo:-nan2088} 

# 克隆 upstream
git clone -b main --depth=1 --recursive --origin upstream git@github.com:torchpipe/torchpipe.git torchpipe

cd torchpipe

# 创建分支
git checkout -b feat/jit

# 添加 fork 作为 origin
git remote add origin "git@github.com:${repo}/torchpipe.git"

# 推送
git push -u origin feat/jit