# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your
# database schema. If you need to create the application database on another
# system, you should be using db:schema:load, not running all the migrations
# from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema.define(version: 2021_06_04_232443) do

  # These are extensions that must be enabled in order to support this database
  enable_extension "plpgsql"

  create_table "history_products", force: :cascade do |t|
    t.bigint "history_record_id"
    t.bigint "product_id"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["history_record_id"], name: "index_history_products_on_history_record_id"
    t.index ["product_id"], name: "index_history_products_on_product_id"
  end

  create_table "history_records", force: :cascade do |t|
    t.string "search_title"
    t.integer "analysis_type"
    t.bigint "user_id"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["user_id"], name: "index_history_records_on_user_id"
  end

  create_table "products", force: :cascade do |t|
    t.string "name"
    t.string "url"
    t.string "image_url"
    t.string "price"
    t.integer "ratings_count"
    t.integer "analyzed_reviews_count"
    t.bigint "supported_website_id"
    t.json "result"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.float "rate"
    t.index ["supported_website_id"], name: "index_products_on_supported_website_id"
  end

  create_table "supported_websites", force: :cascade do |t|
    t.string "name"
    t.string "base_url"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end

  create_table "users", force: :cascade do |t|
    t.string "email", default: "", null: false
    t.string "encrypted_password", default: "", null: false
    t.string "reset_password_token"
    t.datetime "reset_password_sent_at"
    t.datetime "remember_created_at"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["email"], name: "index_users_on_email", unique: true
    t.index ["reset_password_token"], name: "index_users_on_reset_password_token", unique: true
  end

  add_foreign_key "history_products", "history_records"
  add_foreign_key "history_products", "products"
  add_foreign_key "history_records", "users"
  add_foreign_key "products", "supported_websites"
end
