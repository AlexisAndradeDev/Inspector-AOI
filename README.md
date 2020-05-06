# Inspector Alpha



# Instalación


# Instalar módulos necesarios

Descargar el archivo requirements.txt y ejecutar el comando:
    pip install -r requirements.txt
sobre el archivo.


# Instalar paquetes


**Descargar**

* **inspector_package**: Carpeta «inspector_package» ubicada en el repositorio «inspector-alpha».


**Ubicación**

Todos los paquetes (packages) deben estar ubicados en la carpeta Lib ubicada en la carpeta de instalación de Python.
    Python\Python*VERSION*\Lib\

Sustituir *VERSION* por el número de versión de Python, ejemplo: 
    AppData\Local\Programs\Python\Python37\Lib\

Por ejemplo, el paquete **inspector_package** debe ser copiado a la dirección:
    Python\Python*VERSION*\Lib\inspector_package


# Instalar el contenido del repositorio

**IMPORTANTE**: Los paquetes que estén dentro de plibs (como «inspector_package») **no deben ser dejados dentro de plibs**, ya que éstos se instalan en otra dirección.


La carpeta plibs, **después de haber instalado los paquetes que vengan dentro y una vez ya no estén dentro de plibs**, se copiará a la ruta:
    /x64/plibs

Al momento de la instalación, se deben eliminar los archivos *.gitkeep*.

Todos los scripts que sean ejecutados por C# (como ins.py, regallbrds, dbg.py, rt_img.py) deben estar dentro de plibs.
