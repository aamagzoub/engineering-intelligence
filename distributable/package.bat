@echo off
echo ============================================
echo  Packaging Sudanese Wist for distribution
echo ============================================
echo.

cd /d "%~dp0"

:: Clean old game folder
if exist "SudaneseWist\game" rmdir /s /q "SudaneseWist\game"
mkdir "SudaneseWist\game"

:: Copy game code
xcopy /s /e /q "..\gui_pygame" "SudaneseWist\game\gui_pygame\" >nul
xcopy /s /e /q "..\agents" "SudaneseWist\game\agents\" >nul
xcopy /s /e /q "..\environments" "SudaneseWist\game\environments\" >nul
xcopy /s /e /q "..\intelligence" "SudaneseWist\game\intelligence\" >nul

:: Remove __pycache__ folders
for /d /r "SudaneseWist\game" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo.
echo ============================================
echo  Done! Share the "SudaneseWist" folder.
echo  Contents:
echo    setup.bat   - first time setup
echo    play.bat    - launch the game
echo    README.txt  - instructions
echo    game\       - game files
echo ============================================
pause
