cd "/Volumes/DevDereks/dmac90712.github.io" || exit 1
mkdir -p module6
cp /mnt/data/derekmccrarymod6lab1.ipynb module6/
cp /mnt/data/assignments.csv .
git add module6/derekmccrarymod6lab1.ipynb assignments.csv
git commit -m "Add Module 6 notebook and mark assignment as complete"
git push origin main