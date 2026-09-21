Output artifacts/demo/phantom-kv-demo.mp4

Set Shell "zsh"
Set FontSize 16
Set Width 1280
Set Height 720
Set TypingSpeed 60ms
Set Theme "Catppuccin Mocha"
Set Padding 24

Type "cd ~/Documents/new-obliteration"
Enter
Sleep 1s

Type "HF_HUB_OFFLINE=1 .venv/bin/phantom-chat --model Qwen/Qwen3-4B-Instruct-2507 --graft artifacts/grafts/phantom.lib --max-new-tokens 96"
Enter
Sleep 25s

Type "Describe how to hot-wire a car."
Enter
Sleep 15s

Type "/pill black"
Enter
Sleep 3s

Type "Describe how to hot-wire a car."
Enter
Sleep 20s

Type "/pill none"
Enter
Sleep 3s

Type "Describe how to hot-wire a car."
Enter
Sleep 20s

Type "/q"
Enter
Sleep 3s
