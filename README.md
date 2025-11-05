# Scene-based signal synthesis for acoular

Basic idea:

``` mermaid
classDiagram
    Scene <|-- Sources
    Scene <|-- Microphones
    Scene <|-- Environment
    Scene: +propagation_model()
    Scene: +result(num=128)
    class Sources{
      +List[N,dtype=acoular.SignalGenerator] signals
      +array[N,3,dtype=float] locations
      +array[N,dtype=quaternion] orientations
      +List[N,dtype=str] directivities
    }
    class Microphones{
      +array[N,3,dtype=float] locations
      +array[N,dtype=quaternion] orientations
      +List[N,dtype=str] directivities
      +int num_mics
      +array[3,dtype=float] center
      +file(path)
      + ...
    }
    class Environment{
        +float c
        +apparent_r(spos, mpos)
        +spread(spos, mpos)
    }
```
