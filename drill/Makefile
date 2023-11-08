#***************************************************************
#
# メディア情報学実験「音声認識」Makefile
#
# 〜　HMMの基本算法 〜
#
#                                       2013年10月21日 高木一幸
#                                       2023年10月09日 高木一幸
#
#***************************************************************
all: drillF drillB drillT gen drillR 

drillF: drillF.o forward.o
	cc -o drillF drillF.o forward.o

drillF.o: drillF.c
	cc -c -o drillF.o drillF.c

forward.o: forward.c
	cc -c -o forward.o forward.c

drillB: drillB.o backward.o
	cc -o drillB drillB.o backward.o

drillB.o: drillB.c
	cc -c -o drillB.o drillB.c

backward.o: backward.c
	cc -c -o backward.o backward.c

drillT: drillT.o baumwelch.o forward.o backward.o
	cc -o drillT drillT.o baumwelch.o forward.o backward.o -lm

drillT.o: drillT.c
	cc -c -o drillT.o drillT.c

baumwelch.o: baumwelch.c
	cc -c -o baumwelch.o baumwelch.c

gen: gen.c
	cc -o gen gen.c

drillR: drillR.o forward.o
	cc -o drillR drillR.o forward.o -lm

drillR.o: drillR.c
	cc -c -o drillR.o drillR.c
