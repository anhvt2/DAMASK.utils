# reverse of mancasGen_16Oct18.py

i="package3.apdl"
o="input.dat"
rm -v $o

sed -n 24p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 25p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 27p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 28p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 29p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 31p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 32p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 34p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o
sed -n 41p $i  | cut -d"=" -f2 | cut -d"!" -f1 | cut -d"*" -f1 | tr -d " " >> $o
sed -n 42p $i  | cut -d"=" -f2 | cut -d"!" -f1 | cut -d"*" -f1 | tr -d " " >> $o
sed -n 49p $i  | cut -d"=" -f2 | cut -d"!" -f1 | tr -d " " >> $o



