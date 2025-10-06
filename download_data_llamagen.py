import argparse
from huggingface_hub import hf_hub_download
from datasets import load_dataset


def add_arguments(parser: argparse.ArgumentParser):
  parser.add_argument('--data_cache_dir', type=str, 
                      default='./data_cache/imagenet')
  parser.add_argument('--vqgan_cache_dir', type=str, 
                      default='./vqgan_cache/llamagen')  
  parser.add_argument('--hf_token', type=str, default=None)  
  parser.add_argument('--num_proc', type=int, default=1)  


def main():
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()

  if args.hf_token is None:
    raise ValueError('You need to pass a huggingface access token to download Imagenet, with --hf_token')

  print('Downloading ImageNet dataset...')
  load_dataset(
    'timm/imagenet-1k-wds',
    cache_dir=args.data_cache_dir,
    num_proc=args.num_proc,
    token=args.hf_token
  )
  print('ImageNet dataset downloaded successfully!')

  print('Downloading VQGAN checkpoint...')
  hf_hub_download(
      repo_id='FoundationVision/LlamaGen',
      filename='vq_ds8_c2i.pt',
      local_dir=args.vqgan_cache_dir,
  )
  print('VQGAN checkpoint downloaded successfully!')


if __name__ == '__main__':
  main()