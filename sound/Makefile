#***************************************************************
#
# 総合情報学実験メディア「音声認識」Makefile
#
#                                       2013年10月25日 高木一幸
#
#***************************************************************
all: vad _vu ad2txt16

_vu: _vu.c
	cc -o _vu _vu.c -lm

vad: vad.c
	cc -o vad vad.c -lm

ad2txt16: ad2txt16.c
	cc -o ad2txt16 ad2txt16.c -lm
clean:
	\rm _vu vad ad2txt16
