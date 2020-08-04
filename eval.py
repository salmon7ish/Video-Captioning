import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from models_transformer import Transformer as Transformer
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def test(model, crit, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    seq_probs_list = []
    seq_preds_list = []
    masks_list = []
    labels_list = []

    for data in loader:
        # forward the model to get loss
        fc_feats = data['fc_feats'].cuda()
        if(opt["with_mean"] == 0):
                feats_3d = data['feats_3d'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        video_ids = data['video_ids']
      
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if(opt["with_mean"] == 1):
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
            else:
                seq_probs, seq_preds = model(
                    fc_feats, feats_3d, mode='inference', opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]
        
        seq_preds_list.append(seq_preds)
        seq_probs_list.append(seq_probs)
        masks_list.append(masks)
        labels_list.append(labels)

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    seq_probs_list = torch.cat(seq_probs_list, 0)
    seq_preds_list = torch.cat(seq_preds_list, 0)
    labels_list = torch.cat(labels_list, 0)
    masks_list = torch.cat(masks_list, 0)

    return valid_score, samples, seq_probs_list, seq_preds_list, labels_list, masks_list

def save_results(valid_score, samples, opt):
    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results)


def main(opt):
    dataset = VideoDataset(opt, "test")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=opt["bidirectional"],
                             input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                             input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder).cuda()
    elif opt["model"] == "Transformer":
        model = Transformer(opt["vocab_size"], opt["max_len"]).cuda()
    #model = nn.DataParallel(model)
    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    crit = utils.LanguageModelCriterion()

    valid_score, samples, _, _, _, _ = test(model, crit, dataset, dataset.get_vocab(), opt)
    save_results(valid_score, samples, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='',
                        help='path to saved model to evaluate')

    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)
