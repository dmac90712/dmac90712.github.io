cd /mnt/data/python-assignments || exit 1
cp /mnt/data/derekmccrarymod6lab1.ipynb module6/
cp /mnt/data/assignments.csv .
git add module6/derekmccrarymod6lab1.ipynb assignments.csv
git commit -m "Mark Module 6 as complete and add derekmccrarymod6lab1.ipynb"
git push origin main