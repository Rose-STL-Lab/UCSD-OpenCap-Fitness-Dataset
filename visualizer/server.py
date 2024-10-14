import os 
import paramiko
from scp import SCPClient

import bitte_config as bc 
# import options.option_limo as option_limo

class NorthUCSDServer(paramiko.SSHClient):
    def __init__(self, *args, **kwargs):
        key_filename = kwargs['key_filename']
        del kwargs['key_filename']

        super().__init__(*args, **kwargs) 

        self.load_system_host_keys()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        key = paramiko.RSAKey.from_private_key_file(key_filename) # Convert OpenSSH private key is RSA private key: ssh-keygen -p -m PEM -f <path-to-key>  
        self.connect("north.ucsd.edu", port=12700, username="ubuntu", pkey=key,look_for_keys=False,allow_agent=False)    

        self.scp_client = SCPClient(self.get_transport())

    # def state()

    def sync_experiments(self):
        stdin, stdout, stderr = self.exec_command('tree -if --noreport /data/panini/opencap-processing/Data') # -i removes the indentation lines -f returns the full path list --no-report removes directroy and 
        online_experimenets = stdout.readlines()


        with open(os.path.join(bc.SERVER_DIR,"NorthServer_files.txt"),'w') as f: 
            f.write("".join(online_experimenets))



    ################# MAKE SURE ONLY ACCESS ELEMENTS IN THE ENV ##################################

    



class YadiUCSDServer(paramiko.SSHClient):
    def __init__(self, *args, **kwargs):
        key_filename = kwargs['key_filename']
        del kwargs['key_filename']

        super().__init__(*args, **kwargs) 

        self.load_system_host_keys()
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        key = paramiko.RSAKey.from_private_key_file(key_filename) # Convert OpenSSH private key is RSA private key: ssh-keygen -p -m PEM -f <path-to-key>  
        self.connect("roselab1.ucsd.edu", port=40000, username="ubuntu", pkey=key,look_for_keys=False,allow_agent=False)    

        self.scp_client = SCPClient(self.get_transport())

    # def state()

    def sync_experiments(self):
        stdin, stdout, stderr = self.exec_command('tree -if --noreport /data/panini/opencap-processing/Data') # -i removes the indentation lines -f returns the full path list --no-report removes directroy and 
        online_experimenets = stdout.readlines()


        with open(os.path.join(bc.BITTE_DIR,"YadiServer_files.txt"),'w') as f: 
            f.write("".join(online_experimenets))       


        
    # def close(self):
    #     self.close()

if __name__ == "__main__": 
    
    import options_database_sync
    args = options_database_sync.parse_database_sync_options()
    
    # args = option_limo.get_args_parser()
    # torch.manual_seed(args.seed)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark=False


    # args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    # os.makedirs(args.out_dir, exist_ok = True)

    server = NorthUCSDServer(key_filename=args.source_ssh_key)
    success =  server.sync_experiments()

    server.close()