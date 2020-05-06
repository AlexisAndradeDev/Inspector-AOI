# inspector-alpha

Repositorio de código del sistema de inspección Inspector, modelo Alpha.

# Instalar módulos necesarios

Descargar el archivo requirements.txt y ejecutar el comando:
    pip install -r requirements.txt
sobre el archivo.

# Directorios

**Paquetes**

Todos los paquetes (packages) deben estar ubicados en la carpeta Lib ubicada en la carpeta de instalación de Python.
Python\Python*VERSION*\Lib\

Por ejemplo, la carpeta **inspector_package** debe ser copiada a la dirección:
Python\Python*VERSION*\Lib\inspector_package

Sustituir *VERSION* por el número de versión de Python, ejemplo: AppData\Local\Programs\Python\Python37\Lib\inspector_package

**Scripts ejecutados por C#**

Todos los scripts que sean ejecutados por C# (como ins.py, regallbrds, dbg.py, rt_img.py) deben estar ubicados en:
    x64\plibs\

