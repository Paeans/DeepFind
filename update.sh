#!/bin/bash

msg="master update"
[[ $1 == "" ]] || msg=$1

#remotelist="origin bucket"
remotelist[0]=origin
remotelist[1]=bucket
branch="master"

for remote in ${remotelist[@]}
do
    git pull  --no-edit --no-commit $remote $branch
    [[ $? == 0 ]] || {
        echo "ERROR between local and $remote"
        echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        echo
        exit 1
    }
    echo "**************************************"
done
echo

git add --all
git commit -m "${msg}" #&& {
echo "**************************************"
echo

echotag=0
pushtag=1
for remote in ${remotelist[@]}
do
    git push $remote $branch |& grep -v "up-to-date" 
    git push $remote $branch |& while read -r outputstr; do
      [[ $outputstr == "Everything up-to-date" ]] && pushtag=0
    done
    [[ $pushtag == 1 ]]&& {    
      echo "**************************************"
      echotag=1
    }
    #echo
done #} # || echo
[[ $echotag == 1 ]] && echo
