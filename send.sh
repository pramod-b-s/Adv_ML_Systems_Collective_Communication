while IFS= read -r dest; do
  scp recursive_allreduce.py "$dest:~/"
done <destfile.txt
