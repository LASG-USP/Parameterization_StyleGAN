if not DEFINED IS_MINIMIZED set IS_MINIMIZED=1 && start "member_$$" /min "%~dpnx0" %* && exit
ECHO OFF
color 0A

::START /MIN /B /WAIT cmd.exe /c "C:\Program Files (x86)\CMG\IMEX\$version$.10\Win_x64\EXE\mx$version$10.exe" -f member_$$.dat -dd
::START /MIN /B /WAIT cmd.exe /c "C:\Program Files (x86)\CMG\BR\$version$.10\Win_x64\EXE\report.exe" -f member_$$.rwd -o member_$$.rwo
START /MIN /B /WAIT cmd.exe /c "C:\Program Files\CMG\IMEX\$version$.10\Win_x64\EXE\mx$version$10.exe" -f member_$$.dat -dd -wait
START /MIN /B /WAIT cmd.exe /c "C:/Program Files/CMG/RESULTS/$version$.10/Win_x64/exe/report.exe" -f member_$$.rwd -o member_$$.rwo -q 0

del member_$$.out
del member_$$.rwd
del member_$$.mrf
del member_$$.irf
del member_$$.rstr.mrf
del member_$$.rstr.irf
del member_$$.rst
del member_$$.sr3

exit
