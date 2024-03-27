# Upsonic | Self-Driven Autonomous Python Libraries

The Upsonic is designed to help data scientists and ML engineers efficiently manage and automate maintenance-free utility library creation. It provides a simple, easy-to-use Python interface to interact with the Upsonic platform.
[Website](https://upsonic.co/) | [Discord](https://discord.gg/) | [Twitter](https://twitter.com/upsonicco)



## Features

- Easy serialization of functions and classes, making them readily available for reuse across different projects.
- Automatic documentation generation for effortless maintenance and readability.
- Support for both direct and modular function importation from the library.
- Streamlined version control and collaboration features, allowing teams to work together seamlessly.

### Easiest Library View
Usponic proveides an dashboard for your team members. Everyone can access to the dashboard by their [user status](https://docs.upsonic.co/on-prem/using/users). After the accessing they can easily view the top libraries and automaticaly generated connections codes.
![image](https://github.com/Upsonic/Upsonic/assets/41792982/aa67f1f9-e510-4c5f-98fd-6876016157e7)


### Automaticaly Documentation
In Upsonic On-Prem dashboard we have automaticaly generated documentation for your each function, class, object or variables. For this you can use OpenAI GPT integration or a self-hosted Google Gemma model in your installation. They are making your documentations automaticaly. Also you can easily search your content.

- Documentation
- Time Complexity
- Mistakes
- Required Test Tyoes
- Security Analyses
- Tags

![image](https://github.com/Upsonic/Upsonic/assets/41792982/031678af-f0a4-43e9-976b-81707060e85e)

  
## Installation

You need to install the Upsonic container.

[Installing and Running On-Prem Container](https://docs.upsonic.co/on-prem/getting_started/install_on_prem)

Once the container is up and running, you can install the Upsonic Python Client Library on your local system using the pip package manager:
```console
# pip install upsonic
```



## Usage

Upsonic streamlines the development and deployment of utility libraries. Here's how to get started:

### Basic Usage

1. Initialize Upsonic with your server's address:
   ```python
   from upsonic import Upsonic_On_Prem
   upsonic = Upsonic_On_Prem('https://your-server-address:5000', 'ACK_****************')
   ```

2. Define a function you wish to serialize and share:
   ```python
   def sum(a, b):
       return a + b
   ```

3. Serialize the function for Upsonic:
   ```python
   upsonic.dump("math.basics.sum", sum)
   ```

4. Import and use the serialized function in another project:
   ```python
   math = upsonic.load_module("math")
   math.basics.sum(5, 2)
   ```

### Advanced Usage

For complex scenarios and advanced usage, refer to our [detailed documentation](https://docs.upsonic.co/home).

```python
from upsonic import Upsonic_On_Prem
upsonic = Upsonic_On_Prem('https://your-server-address:5000', 'ACK_****************')



def sum(a, b):
    return a + b

upsonic.dump("math.basics.sum", sum)



math = upsonic.load_module("math")

math.basics.sum(5, 2)
```





## Documentation

You can find detailed documentation, including advanced usage and API reference, in the official [Upsonic Documentation](https://docs.upsonic.co/home) .



## Contributing

We welcome contributions to the Upsonic Python Client Library! 



## Support & Questions

For any questions or if you encounter an issue, please reach out to our support team at info@upsonic.co or open an issue on the project's GitHub page.



## Advisors



<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/Upsonic/assets/41792982/b4f4ad18-796e-46ce-9c5a-a50ebaf7f2fe" width="100px;" alt="Bugra Kocaturk"/><br /><b>Buğra Kocatürk</b><br /> <sub>AWS Solution Architect </sub><br /><br /> AWS Netherlands <br /><a href="https://www.linkedin.com/in/bugrakocaturk/" title="Linkedin">Linkedin</a></td>
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/Upsonic/assets/41792982/99307f93-4841-4abb-8ae0-e5e25c75e840" width="100px;" alt="Lemi Orhan Engin"/><br /><b>Lemi Orhan Engin</b><br /> <sub>CTO </sub><br /><br /> Craftgate <br /><a href="https://www.linkedin.com/in/lemiorhan/" title="Linkedin">Linkedin</a></td>
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/Upsonic/assets/41792982/f47a040c-3d38-4965-82ac-27ee064cb501" width="100px;" alt="Mehmet Emin Ozturk"/><br /><b>Mehmet Emin Öztürk</b><br /> <sub>Data Team Lead & Kaggle Master  </sub><br /> Trendyol Group <br /><a href="https://www.linkedin.com/in/meminozturk/" title="Linkedin">Linkedin</a></td>
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/Upsonic/assets/41792982/82388171-1bd8-460c-b7f2-322f94b13707" width="100px;" alt="Firat Gonen"/><br /><b>Fırat Gönen</b><br /> <sub>Chief data Officer & Kaggle Grandmaster 3X </sub><br /> Figopara <br /><a href="https://www.linkedin.com/in/ffgonen/" title="Linkedin">Linkedin</a></td>      
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/7c48bc50-802c-4a27-a61b-d7c12be613de" width="100px;" alt="Arda Batuhan Demir"/><br /><b>Arda Batuhan Demir</b><br /> <sub>Senior DevOps Engineer</sub><br /><br /> Lyrebird Studio <br /><a href="https://www.linkedin.com/in/arda-batuhan-demir/" title="Linkedin">Linkedin</a></td>   </tr>   
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/c27c42e1-ac96-409f-a0c8-aa53a6bfc45a" width="100px;" alt="Hasan Ramazan Yurt"/><br /><b>Hasan Ramazan Yurt</b><br /> <sub>ML Engineer & Technical Founder </sub><br /> Nicky ai <br /><a href="https://www.linkedin.com/in/hryurt/" title="Linkedin">Linkedin</a></td>   
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/ad85d012-cf76-4fb1-a523-b1474139cd38" width="100px;" alt="Sezer Yavuzer Bozkır"/><br /><b>Sezer Yavuzer Bozkır</b><br /> <sub>Sr. Python Developer </sub><br /><br /> Petleo <br /><a href="https://www.linkedin.com/in/sezerbozkir/" title="Linkedin">Linkedin</a></td>        
      <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/13e9f673-88cd-4190-b7e3-9134fe0a4f50" width="100px;" alt="Ozan Günceler"/><br /><b>Ozan Günceler</b><br /> <sub>CTO </sub><br /> BSM Consultancy Limited <br /><a href="https://www.linkedin.com/in/ozangunceler/" title="Linkedin">Linkedin</a></td>         
    <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/cfe14741-42c2-4343-892d-f7654f1e9994" width="100px;" alt="Mustafa Namoğlu"/><br /><b>Mustafa Namoğlu</b><br /> <sub>Co-Founder </sub><br /><br /> İkas <br /><a href="https://www.linkedin.com/in/namoglu/" title="Linkedin">Linkedin</a></td>
    <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/98ff9855-b245-4b7b-8763-90bd9f45bceb" width="100px;" alt="Bünyamin Ergen"/><br /><b>Bünyamin Ergen</b><br /> <sub>AI Engineer & Python Developer & Top Ai Voice </sub><br /> eTaşın <br /><a href="https://www.linkedin.com/in/bunyaminergen/" title="Linkedin">Linkedin</a></td>     </tr>     
    <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/657025f3-23e3-4c4b-8753-248378fdfd9e" width="100px;" alt="Serdar İlarslan"/><br /><b>Serdar İlarslan</b><br /> <sub>Sr. Python developer </sub><br /><br /> Easysize <br /><a href="https://www.linkedin.com/in/serdarilarslan/" title="Linkedin">Linkedin</a></td>         
    <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/2a0d11ea-9185-4da5-8a5c-990b7f58ce2f" width="100px;" alt="Burak Emre Kabakçı"/><br /><b>Burak Emre Kabakçı</b><br /> <sub>Sr. Staff Software Engineer & Maker </sub><br /> LiveRamp <br /><a href="https://www.linkedin.com/in/burak-emre-kabakc%C4%B1-15b2bb33/" title="Linkedin">Linkedin</a></td> 
    <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/0de9cc2d-1f86-46ce-9533-46688c1c836a" width="100px;" alt="Ozge Oz"/><br /><b>Ozge Oz</b><br /> <sub>Partner </sub><br /><br /> QNBEYOND Ventures <br /><a href="https://www.linkedin.com/in/ozge-oz/" title="Linkedin">Linkedin</a></td>   
    <td align="center" valign="top" width="14.28%"><img src="https://github.com/Upsonic/.github/assets/41792982/cc4baafd-973b-490a-91fd-1e2caeeded32" width="100px;" alt="Emre Keskin"/><br /><b>Emre Keskin</b><br /> <sub>Staff Software Engineer </sub><br /><br /> Oplog <br /><a href="https://www.linkedin.com/in/emrekesk-in/" title="Linkedin">Linkedin</a></td>     
    </tr>    
  </tbody>
</table>


## Customers



<table>
   <tr>
      <td><a href="https://wears.com.tr/" target="_blank"><img src="https://github.com/Upsonic/Upsonic/assets/41792982/19f39eec-5251-45ce-837b-8e9db1df17ad" alt="wears" width = 360px></a></td>
       <td><a href="https://exar.com.tr/" target="_blank"><img src="https://github.com/Upsonic/Upsonic/assets/41792982/cf7d9753-97dc-4b9d-9847-4807fdbdd0e3" alt="exar" width = 360px></a></td>
  </tr>
</table>
