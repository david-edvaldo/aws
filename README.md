### Resumo
Este é um conjunto de classes e funções para carregar e salvar dados de diferentes formatos **_('parquet', 'excel', 'csv', 'json', 'pickle', 'image')_** do Amazon S3.

### Dependências
Para usar essas funções, as seguintes dependências devem ser instaladas no ambiente conforme o arquivo **_requirements.txt_**:

### Como Usar
Para usar essas funções, você deve ter **_credenciais de acesso do Amazon S3_** e a permissão para acessar o bucket especificado no código.

### Configurando as Credenciais
O objeto EnvConfig define as credenciais do AWS e é inicializado com um dicionário que contém as seguintes chaves:

**access_key**: _string que representa a chave de acesso do AWS._
**secret_key**: _string que representa a chave secreta do AWS._
**bucket_name**: _string que representa o nome do bucket S3._
**region_name**: _string que representa a região padrão quando são criadas novas conexões._

### Carregando Dados
A classe **UtilitiesS3** possui duas funções chamadas "load_folder" e "load_file" que pode ser usada para carregar dados de diferentes formatos. Os formatos de arquivo suportados são: **_parquet, excel, csv, json, pickle e imagem_**. Você pode carregar os dados usando o nome do arquivo específico ou o nome da pasta onde encontra todos os arquivos que deseja carregar.

Para carregar os dados, você deve seguir o exemplo:

1. Carregar um ou mais arquivo:

![image](https://user-images.githubusercontent.com/78990428/226076121-f9ef55af-dc07-4458-a335-050191f71233.png)

* No dicionário **conn**, deverá conter as credencias para acesso a AWS;
* No dicionário **dataset_info**, deverá conter:
1.1 - O nome do arquivo;
1.2 - 'path': endereço onde se encontra o arquivo no bucket S3,
1.3 - 'pandas_args': são argumentos para utilização das bibliotecas de leituras do pandas (read_excel, read_csv, read_parquet);
1.4 - 'schema': se houver a necessidade de fazer uma conversão dos tipos das variáveis de um DataFrame específico.

* Na chamada da função, será necessário instância a classe **UtilitiesS3**, com os respectivos dicionários.
* Em seguida pode ser chamado o métodos indicado "load_file", no mesmo você pode passar uma lista com os nomes dos arquivos que se deseja carregar no atributo "file_name", os mesmos deverão estar no dicionário **dataset_info**.
* O retorno do método será um dicionário contendo todos os arquivos listado na variável "file_name" do método.

2. Carregar todos os arquivos dentro de uma pasta:
* Você pode usar o mesmo objeto instanciado anteriormente, ou instância um o novo com somente as credencias da AWS.

![image](https://user-images.githubusercontent.com/78990428/226076974-0f68be66-8742-473e-84ec-533ac6e6def5.png)

* Passando para o método o atributo "base_uri", onde está armazenado todos os arquivos que deseja ser carregado.
* O retorno do método será um dicionário.

3. Salvar arquivo:
* Você pode usar o mesmo objeto instanciado anteriormente, passa-se o método chamado "save_file", onde mesmo recebe:
3.1 - obj : O objeto a ser salvo. Deve ser "escolher".
3.2 - file_name   : Nome do arquivo para salvar o objeto, o nome deverá estar no dicionário **dataset_info**..
3.3 - cast_schema : Opção para converter e filtrar o conjunto de dados para o esquema especificado. (Opcional)
3.4 - pandas_args : Argumentos extras para passar para a função salvar os arquivos. (Opcional)

![image](https://user-images.githubusercontent.com/78990428/226077498-0456a0a9-0c17-44c5-b0e4-a6425df1a50d.png)