for dir in $(find . -maxdepth 2 -type d); do
  case "$dir" in
    *#Uce74#Uba54#Ub77c*)
      last_two_chars=$(basename "$dir" | tr -d ' ' | tail -c 3)
      mv "$dir" $(dirname "$dir")/카메라$last_two_chars
      ;;
  esac
done
