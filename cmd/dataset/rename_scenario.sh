for f in *; do
  if [[ $f =~ "#Uc2dc#Ub098#Ub9ac#Uc624*" ]]; then
    mv $f "시나리오${f:(-2)}"
  fi
done

