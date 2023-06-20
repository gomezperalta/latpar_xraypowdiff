#!/bin/bash
#echo $1
content=$(tail -5 $1)
name=$(echo $2 | cut -d '.' -f 1)

mkdir $name
for line in $content; 
  do

  #echo $line;
  z=$(echo $line | cut -d ',' -f 1);

  a=$(echo $line | cut -d ',' -f 2);
  b=$(echo $line | cut -d ',' -f 3);
  c=$(echo $line | cut -d ',' -f 4);

  amin=$(echo $line | cut -d ',' -f 5);
  bmin=$(echo $line | cut -d ',' -f 6);
  cmin=$(echo $line | cut -d ',' -f 7);

  amax=$(echo $line | cut -d ',' -f 8);
  bmax=$(echo $line | cut -d ',' -f 9);
  cmax=$(echo $line | cut -d ',' -f 10);

  v=$(echo $line | cut -d ',' -f 11);
  vmin=$(echo $line | cut -d ',' -f 12);
  vmax=$(echo $line | cut -d ',' -f 13);
  serie=$(echo $name$z)
  #echo $z, $a, $b, $c, $v, $amin, $bmin, $cmin, $vmin, $amax, $bmax, $cmax, $vmax;
  cat template_triclinic.inp | sed 's/diffraction_pattern/'$2$'/g' | sed 's/material_name/'$name$'/g' | sed 's/a0/'$a$'/g' | sed 's/b0/'$b$'/g' | sed 's/c0/'$c$'/g' | sed 's/v0/'$v$'/g' | sed 's/amin/'$amin$'/g' | sed 's/bmin/'$bmin$'/g' | sed 's/cmin/'$cmin$'/g'| sed 's/vmin/'$vmin$'/g'| sed 's/amax/'$amax$'/g'| sed 's/bmax/'$bmax$'/g'| sed 's/cmax/'$cmax$'/g'| sed 's/vmax/'$vmax$'/g' >> triclinic_$serie.inp;
  
  cat template_monoclinic_a.inp | sed 's/diffraction_pattern/'$2$'/g' | sed 's/material_name/'$name$'/g' | sed 's/a0/'$a$'/g' | sed 's/b0/'$b$'/g' | sed 's/c0/'$c$'/g' | sed 's/v0/'$v$'/g' | sed 's/amin/'$amin$'/g' | sed 's/bmin/'$bmin$'/g' | sed 's/cmin/'$cmin$'/g'| sed 's/vmin/'$vmin$'/g'| sed 's/amax/'$amax$'/g'| sed 's/bmax/'$bmax$'/g'| sed 's/cmax/'$cmax$'/g'| sed 's/vmax/'$vmax$'/g' >> monoclinica_$serie.inp;
  cat template_monoclinic_b.inp | sed 's/diffraction_pattern/'$2$'/g' | sed 's/material_name/'$name$'/g' | sed 's/a0/'$a$'/g' | sed 's/b0/'$b$'/g' | sed 's/c0/'$c$'/g' | sed 's/v0/'$v$'/g' | sed 's/amin/'$amin$'/g' | sed 's/bmin/'$bmin$'/g' | sed 's/cmin/'$cmin$'/g'| sed 's/vmin/'$vmin$'/g'| sed 's/amax/'$amax$'/g'| sed 's/bmax/'$bmax$'/g'| sed 's/cmax/'$cmax$'/g'| sed 's/vmax/'$vmax$'/g' >> monoclinicb_$serie.inp;
  cat template_monoclinic_c.inp | sed 's/diffraction_pattern/'$2$'/g' | sed 's/material_name/'$name$'/g' | sed 's/a0/'$a$'/g' | sed 's/b0/'$b$'/g' | sed 's/c0/'$c$'/g' | sed 's/v0/'$v$'/g' | sed 's/amin/'$amin$'/g' | sed 's/bmin/'$bmin$'/g' | sed 's/cmin/'$cmin$'/g'| sed 's/vmin/'$vmin$'/g'| sed 's/amax/'$amax$'/g'| sed 's/bmax/'$bmax$'/g'| sed 's/cmax/'$cmax$'/g'| sed 's/vmax/'$vmax$'/g' >> monoclinicc_$serie.inp;
  
  cat template_orthorhombic.inp | sed 's/diffraction_pattern/'$2$'/g' | sed 's/material_name/'$name$'/g' | sed 's/a0/'$a$'/g' | sed 's/b0/'$b$'/g' | sed 's/c0/'$c$'/g' | sed 's/v0/'$v$'/g' | sed 's/amin/'$amin$'/g' | sed 's/bmin/'$bmin$'/g' | sed 's/cmin/'$cmin$'/g'| sed 's/vmin/'$vmin$'/g'| sed 's/amax/'$amax$'/g'| sed 's/bmax/'$bmax$'/g'| sed 's/cmax/'$cmax$'/g'| sed 's/vmax/'$vmax$'/g' >> orthorhombic_$serie.inp;
done
cp $2 $name/
mv *$name*.inp $name/

