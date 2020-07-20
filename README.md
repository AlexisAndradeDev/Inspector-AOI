# Inspector Alpha



# Instalación


# Carpetas que deben existir antes de la instalación
Dentro de la carpeta de instalación de la versión de Inspector que se tenga (ej. Alpha-Premium) se encuentra el directorio «x64».
Las carpetas que contendrá x64 antes de la instalación son (sin contar el debugeo interactivo, cuya documentación se encuentra en su repositorio https://gitlab.com/Alexismtzan/inspector-interactive-debug):

    x64
        inspections
            data
            status
            bad_windows_results
        plibs
        


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

Todos los scripts que sean ejecutados por C# (como ins.py, regallbrds.py, dbg.py, calregdat.py) deben estar dentro de plibs, y también los «scripts herramienta» como calregdat.py o rt_img.py, que se ejecutan desde C# para hacer algunos cálculos. 

Es decir, **todos los scripts que estén dentro de la carpeta plibs del repositorio y no estén dentro de un paquete, se mantendrán dentro de plibs al momento de la instalación**.
